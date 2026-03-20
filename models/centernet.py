import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# =========================
#   CenterNet Focal Loss
# =========================
def centernet_focal_loss(pred, gt, alpha=2.0, beta=4.0):
    """
    pred: (B,1,H,W) sigmoid outputs
    gt:   (B,1,H,W) heatmap target in [0,1]
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, beta)

    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.sum()

    if num_pos == 0:
        return neg_loss.sum()
    else:
        return (pos_loss.sum() + neg_loss.sum()) / num_pos


# =========================
#   Gaussian Kernel
# =========================
def make_gaussian_kernel(sigma, device="cpu"):
    size = int(6 * sigma + 1)
    coords = torch.arange(size, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    kernel = kernel / kernel.max()  # peak = 1
    return kernel


# =========================
#   Points → CenterNet Heatmap (MAX composition)
# =========================
def points_to_heatmap(points, H, W, sigma, device="cpu"):
    """
    points: Nx2 tensor or array of (y,x)
    returns: (1,1,H,W)
    """
    heatmap = torch.zeros((1, 1, H, W), device=device)

    if len(points) == 0:
        return heatmap

    if not torch.is_tensor(points):
        points = torch.tensor(points, dtype=torch.float32, device=device)

    kernel = make_gaussian_kernel(sigma, device=device)
    k = kernel.shape[0]
    radius = k // 2

    for p in points:
        y, x = int(p[0]), int(p[1])

        if y < 0 or y >= H or x < 0 or x >= W:
            continue

        y0 = max(0, y - radius)
        y1 = min(H, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(W, x + radius + 1)

        ky0 = radius - (y - y0)
        ky1 = radius + (y1 - y)
        kx0 = radius - (x - x0)
        kx1 = radius + (x1 - x)

        heatmap[0, 0, y0:y1, x0:x1] = torch.maximum(
            heatmap[0, 0, y0:y1, x0:x1],
            kernel[ky0:ky1, kx0:kx1]
        )

    return heatmap


# =========================
#   Trainer (CenterNet)
# =========================
class Trainer(object):
    def __init__(self, sigma, device, max_epoch, lr, **kwargs):
        self.sigma = sigma
        self.device = device
        self.lr = lr
        self.threshold = 0.5
        self.nms_kernel_size = 3
        self.max_epoch = max_epoch

    def setup(self, backbone):
        self.device = torch.device(self.device)
        self.backbone = backbone.to(self.device)
        self.optimizer = optim.AdamW(self.backbone.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epoch)
        self.start_epoch = 0

    def train_step(self, inputs, gt_discrete, logger=None):
        inputs = inputs.to(self.device)
        valid = inputs[:, [-1], :, :]  # (B,1,H,W)

        B, _, H, W = inputs.shape

        target_heatmaps = []
        for b in range(B):
            points = torch.nonzero(gt_discrete[b, 0], as_tuple=False)
            heatmap = points_to_heatmap(
                points,
                H,
                W,
                self.sigma,
                device=self.device
            )
            target_heatmaps.append(heatmap)

        target_heatmap = torch.cat(target_heatmaps, dim=0)
        target_heatmap = target_heatmap * valid

        self.backbone.train()

        pred = self.backbone(inputs)
        pred = pred * valid  # mask invalid

        loss = centernet_focal_loss(pred, target_heatmap)

        if logger is not None:
            logger.log({'train/loss': loss.item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        self.backbone.train()

    def eval(self):
        self.backbone.eval()

    def predict(self, inputs):
        self.backbone.eval()
        inputs = inputs.to(self.device)
        valid = inputs[:, [-1], :, :]

        with torch.no_grad():
            hm = self.backbone(inputs)
            hm = hm * valid  # mask invalid
            peak_heatmap = nms(hm, kernel=self.nms_kernel_size)
            pred = peak_heatmap > self.threshold
        return pred * valid

    def hparam_sweep(self, loader):
        self.eval()
        threshs = np.arange(0.02, 1, 0.02)
        preds = {t: [] for t in threshs}
        targets = []
        with torch.no_grad():
            for inp, tgt in loader:
                inp = inp.to(self.device)
                valid = inp[:, [-1], :, :]
                hm = self.backbone(inp)
                hm = hm * valid  # mask invalid
                peak_heatmap = nms(hm, kernel=self.nms_kernel_size)
                for t in threshs:
                    pred = peak_heatmap > t
                    preds[t].extend(pred.view(pred.shape[0], -1).sum(dim=1).cpu())
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

        print(f"Best threshold: {best_thresh:.3f} | nMAE: {best_nmae:.6f}")
        self.threshold = best_thresh
        return

def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

# =========================
#   Peak Extraction
# =========================
def extract_points(heatmap, threshold=0.3):
    """
    heatmap: (1,1,H,W)
    returns: Nx2 (y,x)
    """
    pooled = F.max_pool2d(heatmap, 3, stride=1, padding=1)
    keep = (pooled == heatmap) & (heatmap > threshold)
    return torch.nonzero(keep[0, 0], as_tuple=False)


if __name__ == "__main__":
    from models.backbones import UNetR50
    from data.gf import GFCountingDataset
    from torch.utils.data import DataLoader, random_split
    from models.backbones import UNetR50

    dataset = GFCountingDataset(imsize=64, split="train", preload=False)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)

    backbone = UNetR50(in_channels=4)
    trainer = Trainer(device=torch.device("cpu"), lr=1e-5, sigma=1)
    trainer.setup(backbone)

    for step, (inputs, gt_discrete) in enumerate(loader):
        trainer.hparam_sweep(loader)
        trainer.train_step(inputs, gt_discrete, None)
