import torch
from torch.nn import Module
import torchvision.models as tmodels
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

M_EPS = 1e-16


class Trainer(object):
    def __init__(self, sigma, device, lr, max_epoch, val_epoch, **kwargs):
        self.sigma = sigma
        self.device = device
        self.lr = lr
        self.max_epoch = max_epoch
        self.val_epoch = val_epoch

    def setup(self, backbone):
        self.device = torch.device(self.device)

        self.backbone = backbone.to(self.device)

        self.optimizer = optim.AdamW(self.backbone.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epoch)

        self.start_epoch = 0

        self.loss = nn.MSELoss(reduction="sum").to(self.device)

    def train_step(self, inputs, gt_discrete, logger):
        inputs = inputs.to(self.device)
        valid = inputs[:, [-1,]].to(self.device)

        target_density = []
        points_list = []
        N, _, H, W = inputs.size()
        for b in range(N):
            points = torch.nonzero(gt_discrete[b, 0, :, :], as_tuple=False)
            points_list.append(points.float().to(self.device))
            density = points_to_density(points.cpu().numpy(), H, W, self.sigma, device=self.device)
            #sanity: sum of density should equal number of points
            target_density.append(density)
        target_density = torch.cat(target_density, dim=0).to(self.device) * 100

        with torch.set_grad_enabled(True):
            outputs = nn.functional.softplus(self.backbone(inputs))
            outputs = outputs * valid  # mask invalid regions

            loss = self.loss(outputs, target_density)

            logger.log({
                'train/loss': loss.item(),
            })

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, inputs):
        inputs = inputs.to(self.device)
        valid = inputs[:, [-1,]].to(self.device)
        with torch.no_grad():
            outputs = nn.functional.softplus(self.backbone(inputs)) / 100.0
        return outputs * valid  # mask invalid regions

    def train(self):
        self.backbone.train()

    def eval(self):
        self.backbone.eval()


def points_to_density(points, H, W, sigma, device="cpu"):
    """
    points: list of (y,x) integer coordinates  OR tensor Nx2 (y,x)
    H, W: output height and width
    sigma: Gaussian sigma (in pixels)
    Returns: (1, 1, H, W) density map
    """

    # 1. sparse map with ones at point locations
    target = torch.zeros((1, 1, H, W), device=device)
    if len(points) > 0:
        pts = torch.tensor(points, dtype=torch.long, device=device)
        ys, xs = pts[:, 0], pts[:, 1]
        # mask invalid points
        mask = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        ys, xs = ys[mask], xs[mask]
        target[0, 0, ys, xs] = 1.0

    # 2. Gaussian kernel
    kernel = make_gaussian_kernel(sigma, device=device)
    k = kernel.shape[0]

    # 3. Convolution with padding to keep same size
    density = F.conv2d(
        target,
        kernel.view(1, 1, k, k),
        padding=k // 2
    )

    return density


def make_gaussian_kernel(sigma, device="cpu"):
    """Create 2D Gaussian kernel with size = 6*sigma+1."""
    size = int(6 * sigma + 1)
    coords = torch.arange(size, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = torch.outer(g, g)
    kernel = kernel / kernel.sum()
    return kernel  # (K, K)



if __name__ == "__main__":
    # test OT
    ot_loss = OT_Loss(32, 1, torch.device("cpu"), 100, 10.0)

    dummy_pred = torch.zeros([2, 1, 32, 32])
    points = [torch.tensor([[5.0, 5.0], [10.0, 10.0], [14.0, 14.0]]), torch.tensor([[15.0, 15.0]])]
    dummy_pred[0, 0, 6, 8] = 1.0
    dummy_pred[0, 0, 10, 10] = 1.0
    pred_normed = dummy_pred / (dummy_pred.sum(1).sum(1).sum(1).unsqueeze(1).unsqueeze(1).unsqueeze(1) + 1e-6)
    ot_loss, wd, ot_obj_value = ot_loss(pred_normed, dummy_pred, points)
    print(ot_loss, wd, ot_obj_value)
