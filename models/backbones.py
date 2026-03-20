import torch
import torch.nn as nn
import torchvision.models as models
import timm
import segmentation_models_pytorch as smp
import torch.nn.functional as F


class ResNetCounter(torch.nn.Module):
    def __init__(self, cfg, in_channels=13, pretrained=True):
        super(ResNetCounter, self).__init__()
        self.in_channels = in_channels
        self.backbone = models.resnet50(pretrained=pretrained)
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)  # output single count value

    def forward(self, x):
        return self.backbone(x).squeeze(1)  # return (batch_size,)


class ResNeXtCounter(torch.nn.Module):
    def __init__(self, cfg, in_channels=13, pretrained=True):
        super(ResNeXtCounter, self).__init__()
        self.in_channels = in_channels
        self.backbone = models.resnext50_32x4d(pretrained=pretrained)
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)  # output single count value

    def forward(self, x):
        return self.backbone(x).squeeze(1)  # return (batch_size,)


class SwinT(nn.Module):
    def __init__(self, imsize, in_channels=5):
        super().__init__()

        # Use tiny (lighter, safer for small inputs)
        self.encoder = timm.create_model(
            "swinv2_small_window16_256.ms_in1k",
            pretrained=True,
            features_only=True,
            img_size=imsize,
            out_indices=(2, ),  # 8x down
        )

        # Replace first layer
        old_conv = self.encoder.patch_embed.proj
        new_conv = nn.Conv2d(
            in_channels + 1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        with torch.no_grad():
            new_conv.weight.copy_(
                old_conv.weight.mean(dim=1, keepdim=True)
                .repeat(1, in_channels + 1, 1, 1)
            )
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self.encoder.patch_embed.proj = new_conv

        # Take highest-resolution feature (1/4)
        ch = self.encoder.feature_info.channels()[0]

        self.head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 1),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        feats = self.encoder(x)

        f = feats[0]  # highest resolution (1/4)

        # Swin outputs NHWC → convert
        f = f.permute(0, 3, 1, 2).contiguous()

        out = self.head(f)

        return out


class ViT(nn.Module):
    def __init__(self, imsize, in_channels=13, pretrained=True):
        super().__init__()
        deit = timm.create_model(
            "vit_small_patch8_224", pretrained=pretrained, num_classes=0, img_size=imsize
        )
        # replace patch embedding for in_channels
        conv = deit.patch_embed.proj
        new_conv = nn.Conv2d(
            in_channels + 1,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None
        )
        with torch.no_grad():
            if conv.weight.shape[1] == 3:
                new_conv.weight[:] = conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels + 1, 1, 1)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        deit.patch_embed.proj = new_conv

        self.backbone = deit
        self.patch_size = deit.patch_embed.patch_size[0]
        self.embed_dim = deit.embed_dim
        # linear layer for per-patch prediction
        self.head = nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        B = x.size(0)
        # get patch embeddings
        x = self.backbone.patch_embed(x)  # (B, num_patches, embed_dim)
        # add cls token and pos embedding
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)  # (B, 1+num_patches, embed_dim)

        # remove cls token for per-patch output
        x = x[:, 1:, :]  # (B, num_patches, embed_dim)
        x = self.head(x)  # (B, num_patches, out_channels)

        # reshape to 2D feature map
        H = W = int((x.size(1)) ** 0.5)  # num_patches = H*W
        x = x.transpose(1, 2).reshape(B, 1, H, W)
        return x


class UNetR50(nn.Module):
    def __init__(self, in_channels=4, **kwargs):
        super(UNetR50, self).__init__()
        # SMP loads pretrained weights for encoder only
        encoder_weights = "imagenet"

        # Build U-Net
        self.unet = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=encoder_weights,
            in_channels=in_channels+1,  # override first conv for custom bands
            classes=1,  # output 1 channel
            activation=None  # raw regression output
        )

    def forward(self, x):
        y = self.unet(x)  # (B, 1, H, W)
        return y

    def get_feats(self, x):
        features = self.unet.encoder(x)
        features = self.unet.decoder(features)
        return features


if __name__ == "__main__":
    backbone = SwinT(imsize=64, in_channels=4)
    dummy_input = torch.randn(8, 4, 64, 64)
    out = backbone(dummy_input)
    print(out.shape)