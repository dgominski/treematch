import torch
import torchvision.models as tmodels
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CA_Block(nn.Module):
    """Coordinate Attention block from VrsNet: factorises spatial attention along H and W."""
    def __init__(self, channel, h, w, reduction=16):
        super().__init__()
        self.h = h
        self.w = w
        self.avg_pool_h = nn.AdaptiveAvgPool2d((h, 1))   # avg along W → (B,C,H,1)
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, w))   # avg along H → (B,C,1,W)

        reduced = max(8, channel // reduction)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(channel, reduced, 1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
        )
        self.gate_h = nn.Sequential(nn.Conv2d(reduced, channel, 1, bias=False), nn.Sigmoid())
        self.gate_w = nn.Sequential(nn.Conv2d(reduced, channel, 1, bias=False), nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x_h = self.avg_pool_h(x)                        # (B, C, H, 1)
        x_w = self.avg_pool_w(x).permute(0, 1, 3, 2)   # (B, C, W, 1)
        combined = self.shared_conv(torch.cat([x_h, x_w], dim=2))  # (B, r, H+W, 1)
        h_feat, w_feat = combined.split([self.h, self.w], dim=2)
        return x * self.gate_h(h_feat) * self.gate_w(w_feat).permute(0, 1, 3, 2)


class CountRegressor(nn.Module):
    """Density map decoder from VrsNet. Expects input at H/8 resolution, outputs H."""
    def __init__(self, input_channels=6):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(inplace=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.regressor(x)


class VrsNet(nn.Module):
    """
    ResNet-101 FPN with coordinate attention, adapted for supervised density
    regression without exemplar boxes.

    The ResNet-101 body is frozen; only the input conv, CA blocks, projection
    heads, and CountRegressor are trained.

    map3 (512ch, H/8) and map4 (1024ch, H/16) are attended and projected to
    3ch each, upsampled to the same spatial size, concatenated to 6ch, then
    decoded to a full-resolution density map by CountRegressor.
    """
    def __init__(self, in_channels, imsize):
        super().__init__()
        self.imsize = imsize

        resnet = tmodels.resnet101(pretrained=True)

        # Freeze all ResNet params before extracting sub-modules
        for param in resnet.parameters():
            param.requires_grad = False

        # Adapt input conv for (in_channels + 1) — extra 1 for validity mask
        actual_in = in_channels + 1
        old_conv = resnet.conv1
        self.conv1 = nn.Conv2d(
            actual_in, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            n_copy = min(actual_in, 3)
            self.conv1.weight[:, :n_copy].copy_(old_conv.weight[:, :n_copy])
            if actual_in > 3:
                nn.init.kaiming_normal_(self.conv1.weight[:, 3:])

        self.bn1 = resnet.bn1          # frozen
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1    # 256ch, H/4  — frozen
        self.layer2 = resnet.layer2    # 512ch, H/8  — frozen, map3
        self.layer3 = resnet.layer3    # 1024ch, H/16 — frozen, map4

        h3, h4 = imsize // 8, imsize // 16
        self.ca_map3 = CA_Block(512, h3, h3)
        self.ca_map4 = CA_Block(1024, h4, h4)
        self.proj3 = nn.Conv2d(512, 3, 1, bias=False)
        self.proj4 = nn.Conv2d(1024, 3, 1, bias=False)
        self.regressor = CountRegressor(input_channels=6)

    def trainable_parameters(self):
        return (
            list(self.conv1.parameters())
            + list(self.ca_map3.parameters())
            + list(self.ca_map4.parameters())
            + list(self.proj3.parameters())
            + list(self.proj4.parameters())
            + list(self.regressor.parameters())
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        map3 = self.layer2(x)      # (B, 512, H/8, W/8)
        map4 = self.layer3(map3)   # (B, 1024, H/16, W/16)

        map3 = self.ca_map3(map3)
        map4 = self.ca_map4(map4)

        feat3 = self.proj3(map3)
        feat4 = F.interpolate(self.proj4(map4), size=feat3.shape[2:], mode='bilinear', align_corners=False)
        return self.regressor(torch.cat([feat3, feat4], dim=1))


def make_gaussian_kernel(sigma, device="cpu"):
    size = int(6 * sigma + 1)
    coords = torch.arange(size, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    return kernel / kernel.sum()


def count_map_to_density(count_map, sigma, device="cpu"):
    """
    Convert a rasterized count map to a Gaussian density map.

    Each pixel value n is treated as n trees co-located at that pixel; the map
    is convolved with a unit-sum Gaussian kernel so that the total count is
    preserved (sum of density == sum of count_map).

    Args:
        count_map: (N, 1, H, W) integer or float tensor
        sigma: Gaussian sigma in pixels
        device: target device

    Returns:
        (N, 1, H, W) density map scaled by 100 for numerical stability
    """
    count_map = count_map.float().to(device)
    kernel = make_gaussian_kernel(sigma, device=device)
    k = kernel.shape[0]
    density = F.conv2d(count_map, kernel.view(1, 1, k, k), padding=k // 2)
    return density * 100


class Trainer(object):
    def __init__(self, device, lr, max_epoch, val_epoch, sigma, **kwargs):
        self.device = device
        self.lr = lr
        self.max_epoch = max_epoch
        self.val_epoch = val_epoch
        self.sigma = sigma

    def setup(self, backbone):
        self.device = torch.device(self.device)
        self.backbone = backbone.to(self.device)

        self.optimizer = optim.AdamW(
            self.backbone.trainable_parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epoch)
        self.start_epoch = 0
        self.loss = nn.MSELoss(reduction='none').to(self.device)

    def train_step(self, inputs, valid, gt_discrete, logger):
        inputs = inputs.to(self.device)
        valid = valid.to(self.device)
        target = count_map_to_density(gt_discrete, self.sigma, device=self.device)

        with torch.set_grad_enabled(True):
            outputs = self.backbone(inputs)
            loss = (self.loss(outputs, target) * valid).mean()

            # counts from density: divide out the 100× scale
            pred_count = (outputs.detach() * valid).view(outputs.shape[0], -1).sum(1) / 100
            gt_count = gt_discrete.float().to(self.device).view(gt_discrete.shape[0], -1).sum(1)
            mae = torch.abs(pred_count - gt_count).mean()

            logger.log({
                'train/loss': loss.item(),
                'train/mae': mae.item(),
            })

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, inputs):
        inputs = inputs.to(self.device)
        valid = inputs[:, [-1]].to(self.device)
        with torch.no_grad():
            # divide by 100 to return density in tree-count units
            outputs = self.backbone(inputs) * valid / 100
        return outputs

    def train(self):
        self.backbone.train()

    def eval(self):
        self.backbone.eval()
