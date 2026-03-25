import torch
from torch.nn import Module
import torchvision.models as tmodels
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import datetime
from utils import AverageMeter
import time
from omegaconf import OmegaConf
import ot
from geomloss import SamplesLoss
import matplotlib.pyplot as plt


M_EPS = 1e-16


class Trainer(object):
    def __init__(self, imsize, downscale_ratio, device, wc, wot, reg, reg_m, alpha, num_of_iter_in_ot, lr, clean_ratio, slack, convert_density, max_epoch, **kwargs):
        self.device = device
        self.downscale_ratio = downscale_ratio
        self.wc = wc
        self.wot = wot
        self.reg = reg
        self.reg_m = reg_m
        self.alpha = alpha
        self.imsize = imsize
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.lr = lr
        self.clean_ratio = clean_ratio
        self.max_epoch = max_epoch
        self.slack = slack
        self.convert_density = convert_density

    def setup(self, backbone):
        self.device = torch.device(self.device)

        self.backbone = backbone.to(self.device)

        self.optimizer = optim.AdamW(self.backbone.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epoch)

        self.start_epoch = 0

        self.uot_loss = uOT_Loss(imsize=self.imsize,
                                 downscale_ratio=self.downscale_ratio,
                                device=self.device,
                                num_of_iter_in_ot=self.num_of_iter_in_ot,
                                reg=self.reg,
                                reg_m=self.reg_m,
                                alpha=self.alpha).to(self.device)
        # self.ot_loss = uOT_Loss(imsize=self.imsize,
        #                          device=self.device,
        #                          num_of_iter_in_ot=self.num_of_iter_in_ot,
        #                          reg=self.reg,
        #                          reg_m=None).to(self.device)

        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)

    def train_step(self, inputs, gt_discrete, logger):
        inputs = inputs.to(self.device)
        valid = inputs[:, [-1,]].to(self.device)

        if self.downscale_ratio > 1:
            valid = nn.functional.interpolate(valid.float(), scale_factor=1/self.downscale_ratio, mode="bilinear")

        #convert gt_discrete to points
        points = []
        # densities = []
        for b in range(gt_discrete.size(0)):
            inds = torch.nonzero(gt_discrete[b, 0, :, :], as_tuple=False)
            # density = points_to_density(inds.cpu().numpy(), 64, 64, 2, device=self.device)
            # densities.append(density)
            points.append(inds.float())
        gd_count = np.array([len(p) for p in points], dtype=np.float32)
        points = [p.to(self.device) for p in points]
        gt_discrete = gt_discrete.to(self.device)
        # densities = torch.cat(densities, dim=0).to(self.device)
        N = inputs.size(0)

        clean_batch_size = int(N * self.clean_ratio)

        with torch.set_grad_enabled(True):
            outputs = nn.functional.relu(self.backbone(inputs))
            outputs = outputs * valid  # mask invalid regions

            if clean_batch_size < N:
                # compute OT loss on clean samples
                outputs_clean = outputs[:clean_batch_size]
                points_clean = points[:clean_batch_size]
                gd_count_clean = gd_count[:clean_batch_size]
            else:
                outputs_clean = outputs
                points_clean = points
                gd_count_clean = gd_count

            # Compute uOT loss on clean samples.
            ot_loss = self.uot_loss.forward(outputs_clean, points_clean, slack=False)
            # ot_loss = 0
            # Compute counting loss on clean samples.
            count_loss = self.mae(outputs_clean.sum(1).sum(1).sum(1), torch.from_numpy(gd_count_clean).float().to(self.device)) * self.wc
            # if noisy samples exist, compute UOT loss
            if clean_batch_size < N:
                outputs_noisy = outputs[clean_batch_size:]
                points_noisy = points[clean_batch_size:]
                density_noisy = gt_discrete[clean_batch_size:]

                # uot_loss = self.uot_loss.forward(outputs_noisy, points_noisy, slack=True)
                # plt.imshow(density_noisy[0, 0].cpu().numpy())
                # plt.scatter(x=points_noisy[0][:, 1].cpu().numpy(), y=points_noisy[0][:, 0].cpu().numpy())
                # plt.show()
                if self.convert_density:
                    uot_loss = self.uot_loss.forward_density(outputs_noisy, density_noisy, slack=self.slack)
                else:
                    uot_loss = self.uot_loss.forward(outputs_noisy, points_noisy, slack=self.slack)
                ot_loss = ot_loss + uot_loss
            ot_loss = ot_loss * self.wot

            pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
            mae = np.mean(np.abs(pred_count - gd_count))

            loss = count_loss + ot_loss
            if logger is not None:
                logger.log({
                    'train/total_loss': loss.item(),
                    'train/ot_loss': ot_loss,
                    'train/count_loss': count_loss.item(),
                    'train/mae': mae,
                    # 'train/w_mean': np.mean(w_values),
                    # 'train/w_std': np.std(w_values),
                })

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, inputs):
        inputs = inputs.to(self.device)
        valid = inputs[:, [-1,]].to(self.device)
        if self.downscale_ratio > 1:
            valid = nn.functional.interpolate(valid.float(), scale_factor=1/self.downscale_ratio, mode="bilinear")
        with torch.no_grad():
            outputs = nn.functional.relu(self.backbone(inputs)) * valid
        return outputs

    def train(self):
        self.backbone.train()

    def eval(self):
        self.backbone.eval()


class uOT_Loss(nn.Module):
    """
    - Unbalanced OT (UOT) gives spatial matching
    """

    def __init__(
        self,
        imsize,
        downscale_ratio,
        device,
        num_of_iter_in_ot=100,
        reg=0.1,     # entropy strength ε
        reg_m=1.0,   # KL marginal strength τ (smaller = more unbalanced)
        alpha=0.8 # maximum variation from residuals
    ):
        super().__init__()

        self.imsize = imsize
        self.device = device

        self.loss_fn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=reg ** 0.5,
            reach=reg_m ** 0.5,
            # reach=None,
            debias=False,
            potentials=False,
            backend="tensorized"
        )
        self.uot_potentials = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=reg ** 0.5,
            reach=reg_m ** 0.5,
            # reach=None,
            debias=False,
            potentials=True
        )

        self.num_iter = num_of_iter_in_ot
        self.reg = reg
        self.reg_m = reg_m
        self.alpha = alpha
        self.conf_delay = 400 # number of iterations before introducing confidence weighting
        self.conf_iter = torch.tensor(0)

        # ---- Build coordinate grid (H=W=c_size/stride) ----
        coords = torch.linspace(
            0.5 / (imsize // downscale_ratio), 1 - 0.5 / (imsize // downscale_ratio), (imsize // downscale_ratio)
        )
        Y, X = torch.meshgrid(coords, coords, indexing="xy")

        self.register_buffer(
            "grid",
            torch.stack([X, Y], dim=-1).view(-1, 2).to(device)
        )

    # --------------------------------------
    # Forward
    # --------------------------------------
    def forward(self, unnormed_density, points, slack=True):
        """
        unnormed_density: (B,1,H,W) raw scores
        points: list of length B, each (#gt,2)
        """
        B = unnormed_density.shape[0]
        total = torch.tensor(0.0, device=unnormed_density.device)
        w_values = []

        for b in range(B):
            a = unnormed_density[b, 0].reshape(-1).clamp_min(1e-8)  # (HW,)

            pts = points[b]
            if len(pts) == 0:
                # sinkhorn divergence undefined for empty distribution
                #TODO: or compute tau * sum(a)?
                # total += self.reg_m * a.sum()
                continue

            y = torch.stack([
                pts[:, 0] / self.imsize,
                pts[:, 1] / self.imsize
            ], dim=1)  # (N,2)

            b = torch.ones(len(pts), device=unnormed_density.device)  # (N,)

            # a = a / a.sum()
            # b = b / b.sum()

            # Compute Sinkhorn OT
            if slack:
                b = self.recompute_beta(a, b, self.grid, y)
            # self.plot_ot(a, b, self.grid, y)
            l = self.loss_fn(a, self.grid, b, y)  # (α, x, β, y)
            # if l < 0:
            #     print("negative loss???")
            total += l
        self.conf_iter += 1
        return total

    def forward_density(self, unnormed_density, target_density, slack):
        """
        unnormed_density: (B,1,H,W) raw scores (NOT normalized)
        target_density:   (B,1,H,W) density map (can be normalized)
        """
        B, _, H, W = unnormed_density.shape
        device = unnormed_density.device
        total = torch.tensor(0.0, device=device)

        for b in range(B):
            # source measure (prediction)
            a = unnormed_density[b, 0].reshape(-1).clamp_min(1e-8)

            # target measure (density)
            b = target_density[b, 0].reshape(-1).clamp_min(1e-8)

            # skip empty target
            if b.sum() == 0:
                continue

            # a = a / a.sum()
            # b = b / b.sum()

            # Compute Sinkhorn OT
            if slack:
                b = self.recompute_beta(a, b, self.grid, self.grid)
                total += self.loss_fn(a, self.grid, b, self.grid)  # (α, x, β, y)
            else:
                total += self.loss_fn(a, self.grid, b, self.grid)  # (α, x, β, y)

        return total

    def recompute_beta(self, alpha, beta, x, y):
        with torch.no_grad():
            C = 0.5*(torch.cdist(x, y) ** 2)
            norm_alpha = alpha.detach()
            norm_beta = beta
            F, G = self.uot_potentials(norm_alpha, x, norm_beta, y)
            log_pi = (
                    (F.squeeze().unsqueeze(1) + G.squeeze().unsqueeze(0) - C) / self.reg
                    + torch.log(norm_alpha).unsqueeze(1)
                    + torch.log(norm_beta).unsqueeze(0)
            )
            pi = torch.exp(log_pi)
            # slack = torch.relu(beta - pi.sum(0)) / beta
            residual = pi.sum(0) - norm_beta

            # or without computing transport plan:
            # incoming = beta * torch.exp(-G[0] / self.reg_m)
            # slack = torch.relu(beta - incoming)

            # confidence = self.min_conf + (1 - self.min_conf) * (1 - slack)

            lamb = self.alpha * torch.sigmoid(10 * (self.conf_iter / self.conf_delay - 0.5))
            new_beta = beta + lamb * residual
            # new_beta = new_beta / new_beta.sum()

            # lamb = torch.sigmoid(10 * (self.conf_iter / self.conf_delay - 0.5))
            # w = 1 - lamb + lamb * confidence
        return new_beta

    def plot_ot(self, alpha, beta, x, y):
        with torch.no_grad():
            alpha = alpha.detach().cpu()
            beta = beta.detach().cpu()
            x = x.detach().cpu()
            y = y.detach().cpu()
            C = 0.5*(torch.cdist(x, y) ** 2)

            F, G = self.uot_potentials(alpha, x, beta, y)
            log_pi = (
                    (F[0].unsqueeze(1) + G[0].unsqueeze(0) - C) / self.reg
                    + torch.log(alpha).unsqueeze(1)
                    + torch.log(beta).unsqueeze(0)
            )
            pi = torch.exp(log_pi)
            N = y.shape[0]

            for j in range(N):
                # Mass coming from each pixel to point j
                mass_from_pixels = pi[:, j]  # (D,)
                mass_img = mass_from_pixels.view(self.imsize, self.imsize)

                transported_to_j = mass_from_pixels.sum()

                # Unbalanced residual (mass created at target j)
                created_mass = beta[j] - transported_to_j

                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(alpha.view(self.imsize, self.imsize))
                axs[1].imshow(mass_img)
                axs[1].scatter(
                    y[j, 0] * self.imsize,  # x coordinate (col)
                    y[j, 1] * self.imsize,  # y coordinate (row)
                    c="red",
                    s=40
                )
                plt.title(
                    f"Point {j}\n"
                    f"Transported mass = {transported_to_j:.4f} | "
                    f"Created mass = {created_mass:.4f}"
                )
                plt.show()

        return


def points_to_density(points, H, W, radius, device="cpu"):
    """
    points: list of (y,x) integer coordinates OR tensor Nx2 (y,x)
    H, W: output height and width
    radius: Epanechnikov radius (in pixels)
    Returns: (1, 1, H, W) density map
    """

    # 1. sparse map with ones at point locations
    target = torch.zeros((1, 1, H, W), device=device)
    if len(points) > 0:
        pts = torch.tensor(points, dtype=torch.long, device=device)
        ys, xs = pts[:, 0], pts[:, 1]

        mask = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        ys, xs = ys[mask], xs[mask]

        target[0, 0, ys, xs] = 1.0

    # 2. Epanechnikov kernel
    kernel = make_epanechnikov_kernel(radius, device=device)
    k = kernel.shape[0]

    # 3. Convolution
    density = F.conv2d(
        target,
        kernel.view(1, 1, k, k),
        padding=k // 2
    )

    return density


def make_epanechnikov_kernel(radius, device="cpu"):
    """
    Create 2D Epanechnikov kernel with compact support.
    radius: support radius in pixels
    """

    radius = float(radius)
    size = int(2 * radius + 1)
    coords = torch.arange(size, device=device) - radius

    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r2 = xx**2 + yy**2

    kernel = 1.0 - r2 / (radius**2)
    kernel[r2 > radius**2] = 0.0

    kernel = torch.clamp(kernel, min=0.0)

    # Normalize to sum = 1
    kernel = kernel / kernel.sum()

    return kernel  # (K, K)


def profile_training_step(trainer, loader, batch_size, warmup=10, runs=10):
    # ---- warmup ----
    iterator = iter(loader)
    for _ in range(warmup):
        (inputs, gt_discrete) = next(iterator)
        trainer.train_step(inputs, gt_discrete, None)
    torch.cuda.synchronize()

    # ---- measure time ----
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        (inputs, gt_discrete) = next(iterator)
        trainer.train_step(inputs, gt_discrete, None)
    torch.cuda.synchronize()
    step_time_ms = (time.time() - t0) * 1000 / runs

    # ---- measure FLOPs ----
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True
    ) as prof:
        (inputs, gt_discrete) = next(iterator)
        trainer.train_step(inputs, gt_discrete, None)
        torch.cuda.synchronize()

    flops = sum(e.flops for e in prof.key_averages() if e.flops is not None)
    (inputs, gt_discrete) = next(iterator)
    trainer.train_step(inputs, gt_discrete, None)
    peak_mem = torch.cuda.max_memory_reserved(trainer.device)

    return {
        "time_ms": step_time_ms / batch_size,
        "flops_G": flops / (batch_size * 1e9),
        "memory_MB": peak_mem / (1024**2 * batch_size)
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from models.backbones import UNetR50
    from data.gf import GFPseudoLabelDataset
    from torch.profiler import profile, record_function, ProfilerActivity
    from torch.utils.data import DataLoader, random_split
    import tqdm
    imsize = 64

    device = torch.device("cuda:1")

    backbone = UNetR50()
    backbone.train()

    batch_size = 64
    sizes = [128, 96, 64, 32]
    for s in sizes:
        print(f"imsize == {s}")
        dataset = GFPseudoLabelDataset(imsize=s)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        trainer = Trainer(s, downscale_ratio=1, device=device, wc=1, wot=1, reg=0.005, reg_m=0.2,
                          num_of_iter_in_ot=100, lr=1e-5, clean_ratio=1, slack=True, convert_density=False,
                          max_epoch=100, alpha=0.8)
        trainer.setup(backbone)
        print(profile_training_step(trainer, loader, batch_size))
