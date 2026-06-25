import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.density_regressor import points_to_density


class VarDensBackbone(nn.Module):
    """Wraps UNetR50 to produce (mean, log_var) from shared encoder+decoder features."""
    def __init__(self, backbone):
        super().__init__()
        self.unet = backbone.unet
        dec_channels = self.unet.segmentation_head[0].in_channels  # 16
        self.var_head = nn.Conv2d(dec_channels, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.var_head.weight)
        nn.init.zeros_(self.var_head.bias)

    def forward(self, x):
        enc     = self.unet.encoder(x)
        dec     = self.unet.decoder(enc)
        mean    = self.unet.segmentation_head(dec)  # (B, 1, H, W)
        log_var = self.var_head(dec)                # (B, 1, H, W)
        return mean, log_var


class Trainer(object):
    def __init__(self, sigma, device, lr, max_epoch, val_epoch, **kwargs):
        self.sigma     = sigma
        self.device    = device
        self.lr        = lr
        self.max_epoch = max_epoch
        self.val_epoch = val_epoch

    def setup(self, backbone):
        self.device   = torch.device(self.device)
        self.backbone = VarDensBackbone(backbone).to(self.device)

        self.optimizer = optim.AdamW(self.backbone.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epoch)
        self.start_epoch = 0

        self.loss = nn.GaussianNLLLoss(full=False, reduction='none')

    def train_step(self, inputs, valid, gt_discrete, logger):
        inputs = inputs.to(self.device)
        valid  = inputs[:, [-1,]].to(self.device)

        N, _, H, W = inputs.size()
        target_density = []
        for b in range(N):
            points  = torch.nonzero(gt_discrete[b, 0, :, :], as_tuple=False)
            density = points_to_density(points.cpu().numpy(), H, W, self.sigma, device=self.device)
            target_density.append(density)
        target_density = torch.cat(target_density, dim=0).to(self.device) * 100

        with torch.set_grad_enabled(True):
            mean, log_var = self.backbone(inputs)
            mean = F.softplus(mean) * valid
            var  = F.softplus(log_var).clamp(min=1e-6)

            # GaussianNLL averaged over valid pixels only
            nll  = self.loss(mean, target_density, var)   # (N, 1, H, W)
            loss = (nll * valid).sum() / valid.sum().clamp(min=1)

            logger.log({'train/loss': loss.item()})

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, inputs):
        inputs = inputs.to(self.device)
        valid  = inputs[:, [-1,]].to(self.device)
        with torch.no_grad():
            mean, _ = self.backbone(inputs)
            mean = F.softplus(mean) / 100.0
        return mean * valid

    def train(self):
        self.backbone.train()

    def eval(self):
        self.backbone.eval()
