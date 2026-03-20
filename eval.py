import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt



def game_batched(pred, gt, L):
    """
    pred: (B, 1, H, W) predicted per-pixel counts
    gt:   (B, 1, H, W) ground-truth point maps (0/1 or integer counts)
    L: GAME level (0,1,2,3,...)

    Returns: scalar tensor = average GAME(L) over batch
    """
    B, C, H, W = pred.shape
    n = 2 ** L
    if n > H or n > W:
        raise ValueError(f"2**L = {n} must be <= H and <= W (H={H}, W={W})")
    if H % n != 0 or W % n != 0:
        raise ValueError(f"H and W must be divisible by 2**L. Got H={H}, W={W}, 2**L={n}")

    ph = H // n
    pw = W // n

    # unfold: extracts non-overlapping patches of size (ph, pw) with stride (ph, pw)
    # result shape: (B, C*ph*pw, n*n)
    pred_patches = F.unfold(pred, kernel_size=(ph, pw), stride=(ph, pw))
    gt_patches = F.unfold(gt, kernel_size=(ph, pw), stride=(ph, pw))

    # sum inside each patch: sum over the patch-pixels axis (dim=1) -> (B, n*n)
    pred_counts = pred_patches.sum(dim=1)
    gt_counts = gt_patches.sum(dim=1)

    # reshape counts to (B, n, n) if you want 2D grid, otherwise compute mean directly
    pred_counts = pred_counts.reshape(B, n, n)
    gt_counts = gt_counts.reshape(B, n, n)

    # absolute error per patch, averaged over patches and batch
    return (pred_counts - gt_counts).abs().mean()


def evaluate(ckpt_path, imsize, d):
    from data.gf import GFCountingDataset
    from data.ps import PSCountingDataset
    from data.spot import SPOTCountingDataset
    from models.backbones import UNetR50
    from omegaconf import OmegaConf
    from itertools import product
    from torch.utils.data import DataLoader
    from matplotlib.patches import Circle
    from models.centernet import nms
    from train import get_preds

    datasets = {
        "gf": GFCountingDataset(imsize=imsize, split="test"),
        "ps": PSCountingDataset(imsize=imsize, split="test"),
        "spot": SPOTCountingDataset(imsize=imsize, split="test")
    }

    device = torch.device("cuda:1")
    model = UNetR50(in_channels=4)

    # create loader
    dataset = datasets[d]
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    with torch.no_grad():

        preds = []
        targets = []
        valids = []
        with torch.no_grad():
            for inp, tgt in loader:
                img = inp.to(device)
                valid = inp[:, [-1]]

                output = model(img)
                output = torch.nn.functional.relu(output)
                preds.extend(output.cpu())
                targets.extend(tgt.cpu())
                valids.extend(valid.cpu())

        preds = torch.cat(preds)
        targets = torch.cat(targets)
        valids = torch.cat(valids)
        preds_flat = preds.view(preds.shape[0], -1)
        gts_flat = targets.view(targets.shape[0], -1)
        mask_flat = valids.view(valids.shape[0], -1)

        # Mask the predictions and ground truth
        preds_valid = preds_flat * mask_flat
        gts_valid = gts_flat * mask_flat

        # Sum only over valid regions
        pred_counts = preds_valid.sum(dim=1)
        tgt_counts = gts_valid.sum(dim=1)

        sns.scatterplot(x=tgt_counts, y=pred_counts)
        plt.show()
        # Count the number of valid pixels per sample
        n_valid_pixels = mask_flat.sum(dim=1)
        #ignore empty
        pred_counts = pred_counts[n_valid_pixels > 0]
        tgt_counts = tgt_counts[n_valid_pixels > 0]
        n_valid_pixels = n_valid_pixels[n_valid_pixels > 0]

        # Compute RMSE normalized by number of valid pixels
        squared_error = ((tgt_counts / n_valid_pixels - pred_counts / n_valid_pixels) ** 2)
        rmse = torch.sqrt((squared_error).mean()).item()

        print(rmse)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    ckpts = [
        ("/scratch/checkpoints/treedensity/20260310-0723/", 32),
        ("/scratch/checkpoints/treedensity/20260309-2055/", 64),
        ("/scratch/checkpoints/treedensity/20260309-2144/", 94),
        ("/scratch/checkpoints/treedensity/20260309-2254/", 128)
    ]
    for ckpt_fp, imsz in ckpts:
        evaluate(os.path.join(ckpt_fp, "best_model.pth"), imsz,"ps")

