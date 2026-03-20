import hydra
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import torch
import torchvision.transforms.functional
import tqdm
from torch.utils.data import DataLoader
import os
import rasterio
from itertools import product
from rasterio.windows import Window, bounds, from_bounds
from rasterio.windows import transform as transform_from_window
from shapely.geometry import Point
import geopandas as gpd
import warnings
from utils import normalize_ps, normalize_gf, denormalize_ps, denormalize_gf, denormalize_spot
import glob
import torch.nn as nn

# fix to avoid subprocesses getting stuck due to torch internal CPU multithreading
torch.set_num_threads(1)



def get_tiles(rasterlist, limit):
    tiles = []
    for r in rasterlist:
        rf = rasterio.open(r)
        width = rf.width
        height = rf.height
        nb_divs = max(width // limit, height // limit) + 1
        if nb_divs > 1:
            width_span = width / nb_divs
            height_span = height / nb_divs
            for nb, (i, j) in enumerate(product(range(nb_divs), range(nb_divs))):
                if i == nb_divs and j == nb_divs:
                    window = Window.from_slices(slice(width_span * j, height_span * i), slice(width, height))
                elif i == nb_divs:
                    window = Window.from_slices(slice(width_span * j, height_span * i), slice(width_span * j, height))
                elif j == nb_divs:
                    window = Window.from_slices(slice(width_span * j, height_span * i), slice(width, height_span * i))
                else:
                    window = Window(width_span * j, height_span * i, width=width_span, height=height_span)
                tiles.append([r, window, nb])

        else:
            tiles.append([r, from_bounds(*rf.bounds, rf.transform), None])

    return tiles


@hydra.main(version_base=None, config_path="conf", config_name="viz")
def run_preds(cfg):
    TILE_MAX_SIZE = 20000

    device = torch.device(cfg.device)
    backbone = hydra.utils.instantiate(cfg.backbone)
    ckpt = torch.load(cfg.ckpt)
    backbone.load_state_dict(ckpt)
    backbone.to(device)
    backbone.eval()

    target_images = glob.glob(os.path.join(cfg.target_dir, "*.tif"))
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    tiles = get_tiles(target_images, TILE_MAX_SIZE)

    for t in tiles:
        f, window, window_nb = t

        out_name = os.path.split(f)[1]
        out_fp = os.path.join(cfg.output_dir, os.path.splitext(out_name)[0] + ".tif") # integrate window parameters in filename for merging later
        try:
            with rasterio.open(f) as src:
                raster_array = src.read(window=window)
                profile = src.profile
        except Exception as e:
            warnings.warn(f"Could not read raster {f}, skipping. Error: {e}")
            continue
        raster_array = raster_array.astype(float)
        raster_array = torch.from_numpy(raster_array).unsqueeze(0)

        if cfg.scaling == "ps":
            raster_array = normalize_ps(raster_array)
        elif cfg.scaling == "gf":
            raster_array = normalize_gf(raster_array)
        elif cfg.scaling == "spot":
            raster_array = scale_ls_frame(raster_array)
        else:
            raise ValueError(f"Unknown scaling {cfg.scaling}, please use 'ps' or 'gf'.")
        raster_array = torch.cat([raster_array, torch.ones_like(raster_array)[:, [0]]], dim=1)

        # make tensor dimensions multiples of 8
        x_pad, y_pad = 0, 0
        if raster_array.shape[2] % 8 != 0:
            x_pad = raster_array.shape[2] % 8
            raster_array = torch.nn.functional.pad(raster_array, (0, 0, 0, x_pad), mode='constant', value=0)
        if raster_array.shape[3] % 8 != 0:
            y_pad = raster_array.shape[3] % 8
            raster_array = torch.nn.functional.pad(raster_array, (0, y_pad, 0, 0), mode='constant', value=0)

        with torch.no_grad():
            transformed = raster_array.float()
            probs_batch = backbone(transformed.to(device))

        probs_batch = torch.clamp(probs_batch, min=0, max=1)  # make sure everything is in the 0-255 range

        new_array = np.concatenate([probs_batch.cpu().numpy()[0]], axis=0)
        del probs_batch

        if x_pad:
            new_array = new_array[:, :-x_pad, :]
        if y_pad:
            new_array = new_array[:, :, :-y_pad]

        transform = rasterio.windows.transform(window, profile["transform"])
        profile.update(dtype="float32", count=new_array.shape[0], compress="LZW", nodata=None, driver="GTiff",
                       width=new_array.shape[2], height=new_array.shape[1], transform=transform)

        try:
            with rasterio.open(out_fp, 'w', **profile) as out_ds:
                out_ds.write(new_array)
        except Exception as e:
            warnings.warn(f"Could not save output raster for {out_fp}, skipping. Error: {e}")
            del new_array, t_size, p0, p1, window, profile, out_fp
            continue


@hydra.main(version_base=None, config_path="conf", config_name="viz")
def viz_slack(cfg):
    from models.udm import uOT_Loss
    from matplotlib.colors import TwoSlopeNorm
    device = cfg.device

    out_dir = "/home/dgominski/Documents/publis/tree_density/residuals/spot/"

    # instantiate dataset
    dataset = hydra.utils.instantiate(cfg.dataset_noisy, split="train")
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True)

    # instantiate backbone
    backbone = hydra.utils.instantiate(cfg.backbone)
    ckpt = torch.load(cfg.ckpt)
    backbone.load_state_dict(ckpt)
    backbone.to(device)
    backbone.eval()

    uot = uOT_Loss(imsize=64, device=device, num_of_iter_in_ot=100, reg=0.005, reg_m=0.1, alpha=0.8)
    uot.conf_iter += 1000

    for epoch in range(cfg.train.nepoch):
        for step, (inputs, gt_discrete) in enumerate(loader):
            inputs = inputs.to(device)
            valid = inputs[:, [-1, ]].to(device)

            # convert gt_discrete to points
            points = []
            # densities = []
            for b in range(gt_discrete.size(0)):
                inds = torch.nonzero(gt_discrete[b, 0, :, :], as_tuple=False)
                points.append(inds.float())
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(device) for p in points]
            N = inputs.size(0)

            file_id = 0
            with torch.no_grad():
                outputs = nn.functional.relu(backbone(inputs))

                for i in range(N):
                    a = outputs[i, 0].reshape(-1).clamp_min(1e-8)  # (HW,)

                    pts = points[i]
                    if len(pts) == 0:
                        continue

                    y = torch.stack([
                        pts[:, 0] / 64,
                        pts[:, 1] / 64
                    ], dim=1)  # (N,2)

                    b1 = torch.ones(len(pts), device=outputs.device)  # (N,)
                    b2 = uot.recompute_beta(a, b1, uot.grid, y)


                    # --- Images ---
                    if cfg.scaling == "ps":
                        img = denormalize_ps(inputs[i].cpu())
                    elif cfg.scaling == "gf":
                        img = denormalize_gf(inputs[i].cpu())
                    elif cfg.scaling == "spot":
                        img = denormalize_spot(inputs[i].cpu())
                    else:
                        raise ValueError(f"Unknown scaling {cfg.scaling}, please use 'ps' or 'gf'.")

                    y_np = y.detach().cpu().numpy()
                    rows = y_np[:, 0] * 64
                    cols = y_np[:, 1] * 64
                    b2_np = b2.detach().cpu().numpy() - 1

                    norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)

                    # ---------- 1️⃣ INPUT ----------
                    fig = plt.figure(figsize=(3, 3))
                    ax = fig.add_subplot(111)
                    ax.imshow(img[:3].numpy().transpose(1, 2, 0))
                    ax.axis("off")
                    fig.savefig(os.path.join(out_dir, f"input_{file_id}.png"), dpi=450, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)

                    # ---------- 2️⃣ LABELS ----------
                    fig = plt.figure(figsize=(3, 3))
                    ax = fig.add_subplot(111)
                    ax.scatter(cols, rows, s=10, c="green", edgecolors="black", linewidths=0.5)
                    ax.set_xlim(0, 64)
                    ax.set_ylim(64, 0)
                    ax.set_aspect("equal")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.savefig(os.path.join(out_dir, f"labels_{file_id}.png"), dpi=450, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)

                    # ---------- 3️⃣ PRED ----------
                    fig = plt.figure(figsize=(3, 3))
                    ax = fig.add_subplot(111)
                    ax.imshow(a.view(64, 64).cpu().numpy(), cmap="YlOrBr")
                    ax.axis("off")
                    fig.savefig(os.path.join(out_dir, f"pred_{file_id}.png"), dpi=450, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)

                    # ---------- 4️⃣ RESIDUALS ----------
                    fig = plt.figure(figsize=(3, 3))
                    ax = fig.add_subplot(111)
                    sc = ax.scatter(
                        cols, rows,
                        c=b2_np,
                        cmap="seismic",
                        norm=norm,
                        s=10,
                        edgecolors="black",
                        linewidths=0.5
                    )
                    ax.set_xlim(0, 64)
                    ax.set_ylim(64, 0)
                    ax.set_aspect("equal")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.savefig(os.path.join(out_dir, f"residuals_{file_id}.png"), dpi=450, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)
                    file_id += 1


def compute_residuals():
    from models.udm import SamplesLoss
    from matplotlib.colors import TwoSlopeNorm

    reg = 0.005
    reg_m = 500

    pred_fp = "/home/dgominski/Documents/publis/tree_density/first_fig_retry/pred.tif"
    weak_labels_fp = "/home/dgominski/Documents/publis/tree_density/first_fig_retry/pseudolabels/CHM_crop_0_0.geojson"
    pred_raster = rasterio.open(pred_fp)
    weak_points = gpd.read_file(weak_labels_fp).iloc[:-1]
    x = [p.x for p in weak_points.geometry]
    y = [p.y for p in weak_points.geometry]
    weak_points = np.array(pred_raster.index(x, y)).T
    preds = torch.Tensor(pred_raster.read())[0]

    uot_potentials = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=reg ** 0.5,
        reach=reg_m ** 0.5,
        # reach=None,
        debias=False,
        potentials=True
    )

    x = torch.argwhere(preds > 0).contiguous()
    alpha = preds[x[:, 0], x[:, 1]].contiguous()
    y = torch.Tensor(weak_points).contiguous()
    beta = torch.ones(y.shape[0]).contiguous()

    x = x / preds.shape[1]
    y = y / preds.shape[1]

    with torch.no_grad():
        C = 0.5*(torch.cdist(x, y) ** 2)
        norm_alpha = alpha.detach()
        norm_beta = beta
        F, G = uot_potentials(norm_alpha, x, norm_beta, y)
        log_pi = (
                (F[0].unsqueeze(1) + G[0].unsqueeze(0) - C) / reg
                + torch.log(norm_alpha).unsqueeze(1)
                + torch.log(norm_beta).unsqueeze(0)
        )
        pi = torch.exp(log_pi)
        # slack = torch.relu(beta - pi.sum(0)) / beta
        residual = 1.5 * (pi.sum(0) - norm_beta)
        asd = pi.sum(0).numpy()

    # --- 1️⃣ Convert normalized coordinates back to pixel indices
    rows = (y[:, 0] * preds.shape[1]).numpy()
    cols = (y[:, 1] * preds.shape[1]).numpy()

    # --- 2️⃣ Convert pixel indices to raster CRS coordinates
    xs, ys = rasterio.transform.xy(
        pred_raster.transform,
        rows,
        cols,
        offset="center"
    )

    # --- 3️⃣ Build GeoDataFrame in raster CRS
    gdf = gpd.GeoDataFrame(
        {
            "residual": residual.numpy()
        },
        geometry=[Point(xc, yc) for xc, yc in zip(xs, ys)],
        crs=pred_raster.crs
    )

    # --- 4️⃣ Reproject to WGS84
    gdf_wgs84 = gdf.to_crs(epsg=4326)

    # --- 5️⃣ Save as GeoPackage
    output_fp = "/home/dgominski/Documents/publis/tree_density/first_fig_retry/uot_residuals.gpkg"
    gdf_wgs84.to_file(output_fp, driver="GPKG")


def get_preds(loader, model, device):
    device = torch.device(device)
    preds = []
    targets = []
    with torch.no_grad():
        for inp, tgt in loader:
            inputs = inp.to(device)
            valid = inp[:, [-1, ]].to(device)
            start = time.time()
            with torch.no_grad():
                outputs = nn.functional.relu(model(inputs)) * valid
            end = time.time()
            print(f"time for processing {outputs.numel()} pixels == {end - start}")

            preds.extend(outputs.cpu())
            targets.extend(tgt.cpu())
    return torch.cat(preds), torch.cat(targets)


def evaluate(preds, gts):
    # patch-level
    pred_counts = preds.view(preds.shape[0], -1).sum(dim=1)
    tgt_counts = gts.view(gts.shape[0], -1).sum(dim=1)
    mae = torch.abs(tgt_counts - pred_counts) / tgt_counts
    return mae


@hydra.main(version_base=None, config_path="conf", config_name="viz")
def plot_preds(cfg):
    from data.gf import GFCountingDataset
    from data.ps import PSCountingDataset
    from data.spot import SPOTCountingDataset
    from models.backbones import UNetR50
    from omegaconf import OmegaConf
    from itertools import product
    from torch.utils.data import DataLoader
    from matplotlib.patches import Circle
    from models.centernet import nms

    datasets = {
        "gf": GFCountingDataset(imsize=64, split="test"),
        "ps": PSCountingDataset(imsize=64, split="test"),
        "spot": SPOTCountingDataset(imsize=64, split="test")
    }

    denorm = {
        "ps": denormalize_ps,
        "gf": denormalize_gf,
        "spot": denormalize_spot
    }

    device = torch.device("cuda:1")

    ckpts = OmegaConf.load("conf/ckpt.yaml")

    models = {
        "udm": UNetR50(in_channels=4),
        "dens": UNetR50(in_channels=4),
        "centernet": UNetR50(in_channels=4),
        "dmcount": UNetR50(in_channels=4)
    }

    for d in ["gf"]:
        out_dir = f"/home/dgominski/Documents/publis/tree_density/fig_supp/{d}"

        # create loader
        dataset = datasets[d]
        loader = DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1, pin_memory=True)

        # load models
        for m in models:
            ckpt_path = ckpts[f"{d}_{m}"]
            print(ckpt_path)
            ckpt = torch.load(ckpt_path)
            models[m].load_state_dict(ckpt)
            models[m].to(device)
            models[m].eval()

        if "spot" in d:
            radius = 10
            y, x = np.ogrid[:64, :64]
            dist = np.sqrt((x - 32) ** 2 + (y - 32) ** 2)
            alpha = np.clip(1 - (dist - radius) / 2, 0.1, 1)
        else:
            alpha = np.ones((64, 64))

        with torch.no_grad():
            i = 0

            # run once to find images with lowest mae
            loader_eval = DataLoader(dataset, batch_size=128, shuffle=False,num_workers=4, pin_memory=True)
            preds, gt = get_preds(loader_eval, models["udm"], device=device)
            maes = evaluate(preds, gt)
            best_idxs = torch.argsort(maes, descending=True)

            for idx in best_idxs[::2]:
                (inputs, gt_discrete) = dataset[idx]

                img = inputs.to(device)[None, ]
                valid = inputs[-1]

                if "spot" in d:
                    img[0, -1] = 1

                img_viz = denorm[d](img[0].cpu()).numpy()
                # img_viz = ((img_viz - img_viz.min()) / (img_viz.max() - img_viz.min()) * 255).astype(np.uint8)
                plt.imshow(img_viz[:3].transpose(1, 2, 0))
                plt.axis("off")
                fig = plt.gcf()
                fig.savefig(os.path.join(out_dir, f"{i}_input.png"), dpi=450, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                # convert gt_discrete to points
                points = torch.nonzero(gt_discrete[0, :, :], as_tuple=False)
                rows = points[:, 0]
                cols = points[:, 1]
                fig, ax = plt.subplots()
                ax.scatter(cols, rows, s=250, c="green", edgecolors="black", linewidths=2, marker="+")
                ax.set_xlim(0, 64)
                ax.set_ylim(64, 0)
                ax.set_aspect("equal")
                plt.axis("off")
                if "spot" in d:
                    fig.patch.set_facecolor((0.5, 0.5, 0.5))
                    ax.set_facecolor((0.5, 0.5, 0.5))

                    # --- White central disk ---
                    white = np.ones_like(alpha)
                    ax.imshow(
                        white,
                        cmap="gray",
                        vmin=0,
                        vmax=1,
                        alpha=alpha,  # 1 inside disk → white visible
                        origin="upper"
                    )
                    ax.scatter(
                        cols,
                        rows,
                        c="green",
                        s=250,
                        edgecolors="black",
                        linewidths=2,
                        marker="+"
                    )
                    ax.set_xlim(0, 64)
                    ax.set_ylim(64, 0)
                    ax.set_aspect("equal")

                fig = plt.gcf()
                fig.savefig(os.path.join(out_dir, f"{i}_labels_{rows.shape[0]}.png"), dpi=450, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                for m in models:
                    start = time.time()
                    out = models[m](img)

                    fig, ax = plt.subplots()
                    fig.patch.set_facecolor((0.5, 0.5, 0.5))  # figure background
                    # ax.set_facecolor((0.5, 0.5, 0.5))  # axes background
                    if "dens" in m:
                        out = nn.functional.softplus(out) / 100.0
                        ax.imshow(out[0,0].cpu().numpy(), cmap="YlOrBr", alpha=alpha)
                        preds = np.round((out.cpu() * valid.cpu()).sum().numpy(), decimals=1)

                    elif "centernet" in m:
                        # fig.patch.set_facecolor((1, 1, 1))  # figure background

                        # out = out * valid.to(device)  # mask invalid
                        peak_heatmap = nms(out, kernel=3)
                        pred_for_plot = peak_heatmap > 0.5
                        pred_for_plot = torch.nonzero(pred_for_plot[0, 0, :, :], as_tuple=False).cpu().numpy()

                        preds = (peak_heatmap > 0.5) * valid.to(device)
                        preds = torch.nonzero(preds[0, 0, :, :], as_tuple=False).cpu().numpy()
                        preds = preds.shape[0]

                        fig.patch.set_facecolor((0.5, 0.5, 0.5))
                        ax.set_facecolor((0.5, 0.5, 0.5))

                        # --- White central disk ---
                        white = np.ones_like(alpha)
                        ax.imshow(
                            white,
                            cmap="gray",
                            vmin=0,
                            vmax=1,
                            alpha=alpha,  # 1 inside disk → white visible
                            origin="upper"
                        )

                        # --- Scatter points with transparency outside disk ---
                        rows = pred_for_plot[:, 0].astype(int)
                        cols = pred_for_plot[:, 1].astype(int)
                        inside = alpha[rows, cols] > 0.5

                        colors = np.zeros((len(pred_for_plot), 4))
                        colors[:, 2] = 1.0  # blue channel
                        colors[inside, 3] = 1.0  # fully opaque inside disk
                        colors[~inside, 3] = 0.25  # more transparent outside disk

                        ax.scatter(
                            pred_for_plot[:, 1],
                            pred_for_plot[:, 0],
                            s=250,
                            c=colors,
                            edgecolors="black",
                            linewidths=2,
                            marker="+"
                        )
                        ax.set_xlim(0, 64)
                        ax.set_ylim(64, 0)
                        ax.set_aspect("equal")
                    else:
                        out = nn.functional.relu(out)
                        ax.imshow(out[0,0].cpu().numpy(), cmap="YlOrBr", alpha=alpha)
                        preds = np.round((out.cpu() * valid.cpu()).sum().numpy(), decimals=1)

                    ax.axis("off")


                    fig.savefig(
                        os.path.join(out_dir, f"{i}_{m}_{preds:.1f}.png"),
                        dpi=450,
                        bbox_inches="tight",
                        pad_inches=0
                    )

                    plt.close(fig)
                i += 1





if __name__ == '__main__':
    # predcrop = rasterio.open("/home/dgominski/Documents/publis/tree_density/first_fig_retry/pred.tif").read()
    # print(predcrop.sum())
    # run_preds()
    # viz_slack()
    # compute_residuals()
    plot_preds()