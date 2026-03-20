import torch
from torch.nn import Module
import torchvision.models as tmodels
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import datetime
from utils import AverageMeter
import time
from omegaconf import OmegaConf
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

M_EPS = 1e-16

from torchvision.ops import nms
from torchmetrics.detection import MeanAveragePrecision


class Trainer(object):
    def __init__(self, imsize, device, lr, max_epoch, **kwargs):
        self.device = device
        self.imsize = imsize
        self.lr = lr
        self.threshold = 0.5
        self.max_epoch = max_epoch

    def setup(self, backbone):
        self.device = torch.device(self.device)

        self.backbone = backbone.to(self.device)
        self.head = P2PHead(in_channels=16).to(self.device)

        self.optimizer = optim.AdamW(list(self.backbone.parameters()) + list(self.head.parameters()), lr=self.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epoch)

        self.start_epoch = 0

    def train_step(self, inputs, gt_discrete, logger):
        inputs = inputs.to(self.device)
        valid = inputs[:, [-1,]].to(self.device)

        B, C, H, W = inputs.shape

        #convert gt_discrete to points
        points = []
        for b in range(gt_discrete.size(0)):
            inds = torch.nonzero(gt_discrete[b, 0, :, :], as_tuple=False)
            points.append(inds.float())
        gd_count = np.array([len(p) for p in points], dtype=np.float32)
        gt_points = [p.to(self.device) for p in points]

        with torch.set_grad_enabled(True):
            feats = self.backbone.get_feats(inputs)

            cls = self.head(feats)

            grid = make_grid(H, W, inputs.device)
            loss_cls = p2p_loss(cls, gt_points, grid, valid)
            loss = loss_cls

            pred_count = torch.sum(cls.sigmoid(), dim=1).squeeze(-1).detach().cpu().numpy()
            mae = np.mean(np.abs(pred_count - gd_count))

            if logger is not None:
                logger.log({
                    'train/mae': mae,
                    'train/loss_cls': loss_cls.item(),
                    'train/total_loss': loss.item(),
                })

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self):
        self.backbone.train()
        self.head.train()

    def eval(self):
        self.backbone.eval()
        self.head.eval()

    def predict(self, inputs):
        self.backbone.eval()
        self.head.eval()

        inputs = inputs.to(self.device)
        B, C, H, W = inputs.shape

        with torch.no_grad():
            feats = self.backbone.get_feats(inputs)
            cls = self.head(feats)
            probs = cls.sigmoid()

            grid = make_grid(H, W, inputs.device)
            points_batch = decode_points(probs, grid, H, W, threshold=self.threshold)

        return points_batch

    def hparam_sweep(self, loader):
        self.eval()
        threshs = np.arange(0.02, 1, 0.02)
        preds = {t: [] for t in threshs}
        targets = []
        with torch.no_grad():
            for inp, tgt in loader:
                inp = inp.to(self.device)
                B, C, H, W = inp.shape

                with torch.no_grad():
                    feats = self.backbone.get_feats(inp)
                    cls = self.head(feats)
                    grid = make_grid(H, W, inp.device)
                    for t in threshs:
                        pb = decode_points(cls.sigmoid(), grid, H, W, threshold=t)
                        preds[t].extend(pb.view(pb.shape[0], -1).sum(dim=1).cpu())
                    targets.extend(tgt.view(tgt.shape[0], -1).sum(dim=1))

        targets = np.array(targets)
        mean_gt = targets.mean() + 1e-8  # avoid division by zero

        best_thresh = None
        best_nmae = float("inf")

        for t in threshs:
            pred_arr = np.array(preds[t])
            nmae = np.mean(np.abs(pred_arr - targets)) / mean_gt
            if nmae < best_nmae:
                best_nmae = nmae
                best_thresh = t

        # print(f"Best threshold: {best_thresh:.3f} | nMAE: {best_nmae:.6f}")
        self.threshold = best_thresh
        return


class P2PHead(nn.Module):
    """
    UNet-based P2PNet head
    Input: feature map (B, C, H, W)
    Output:
      - cls: point confidence (B, HW, 1)
    """

    def __init__(self, in_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.cls_head = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.conv(x)

        cls = self.cls_head(x)          # (B,1,H,W)

        cls = cls.view(B, -1, 1)
        return cls


def make_grid(H, W, device):
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    return torch.stack([x, y], dim=-1).view(-1, 2)


def decode_points(probs, grid, H, W, threshold=0.5):
    """
    probs:  (B, HW, 1)
    grid: (HW, 2)
    H, W: spatial size of output map

    Returns:
        out: (B, H, W) binary map
    """

    B = probs.shape[0]

    # Apply classification threshold
    mask = probs[..., 0] > threshold        # (B, HW)

    # Initialize output map
    out = torch.zeros((B, H, W), device=probs.device)

    for b in range(B):
        pts = grid[mask[b]]         # (N, 2)

        if pts.numel() == 0:
            continue

        # Round to nearest integer pixel
        x = pts[:, 0].round().long()
        y = pts[:, 1].round().long()

        # Keep valid coordinates
        valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        x = x[valid]
        y = y[valid]

        out[b, y, x] = 1

    return out[:, None]


def p2p_loss(cls, points, grid, valid_mask,
             alpha=0.25, gamma=2.0):
    """
    cls:        (B, HW, 1)  (logits)
    points:     list of (N,2)
    grid:       (HW, 2)
    valid_mask: (B, HW)
    """

    B, HW, _ = cls.shape

    cls_loss = torch.tensor(0.0, device=cls.device)

    for b in range(B):
        gt = points[b].to(cls.device)
        pred = grid.float()

        valid = valid_mask[b].reshape(-1).bool()
        valid_idx = torch.nonzero(valid).squeeze(1)

        # --------------------------------------------------
        # No GT → all valid pixels are negatives
        # --------------------------------------------------
        if len(gt) == 0:

            logits = cls[b, valid, 0]
            targets = torch.zeros_like(logits)

            cls_loss += focal_loss_binary(
                logits, targets, alpha, gamma
            ).mean()
            continue

        C = torch.cdist(pred, gt)
        row, col = linear_sum_assignment(
            C.detach().cpu().numpy()
        )

        # --------------------------------------------------
        # Classification (focal on valid only)
        # --------------------------------------------------
        logits = cls[b, :, 0]
        targets = torch.zeros_like(logits)
        targets[row] = 1.0

        cl = focal_loss_binary(
            logits, targets, alpha, gamma
        )
        cls_loss += (cl * valid).mean()

    cls_loss = cls_loss / B

    return cls_loss


def focal_loss_binary(logits, targets, alpha=0.25, gamma=2.0):
    """
    logits:  (N,)
    targets: (N,)  0 or 1
    """

    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )

    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    loss = alpha_t * (1 - p_t) ** gamma * ce
    return loss


if __name__ == "__main__":
    from models.backbones import UNetR50
    from data.gf import GFCountingDataset
    from torch.utils.data import DataLoader, random_split
    from models.backbones import UNetR50

    dataset = GFCountingDataset(imsize=64, split="train", preload=False)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)

    backbone = UNetR50(in_channels=4)
    trainer = Trainer(wcls=1, wreg=1, imsize=64, device=torch.device("cpu"), lr=1e-5)
    trainer.setup(backbone)

    for step, (inputs, gt_discrete) in enumerate(loader):
        trainer.train_step(inputs, gt_discrete, None)

