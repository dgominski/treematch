from predict_multi_thread import get_tiles, get_prediction_fps
import rasterio
import os
import numpy as np
import hydra
from models.backbones import UNetCounter
import torch
from utils import split_tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize


TILE_MAX_SIZE = 5000
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="conf", config_name="predict")
def test(cfg):
    model = UNetCounter(cfg, in_channels=20, pretrained=False)
    ckpt = torch.load(cfg.ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()

    band_stats = np.load("data/ps_band_stats.npz")
    normalizer = Normalize(mean=band_stats['mean'].tolist(), std=band_stats['std'].tolist())

    in_fps = get_prediction_fps(cfg)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    target_images = np.array(in_fps).tolist()

    ps_fps = [ti[0] for ti in target_images]
    s2_fps = [ti[1] for ti in target_images]
    s1_fps = [ti[2] for ti in target_images]
    tiles = get_tiles(ps_fps, s2_fps, s1_fps, limit=TILE_MAX_SIZE)

    for ps_fp, s2_fp, s1_fp, window, window_nb in tiles:
        out_fp = os.path.join(cfg.output_dir, f"{os.path.split(ps_fp)[-1].replace('.jp2', '')}_{window_nb}.tif")
        if os.path.exists(out_fp):
            print(f"Skipping existing file {out_fp}")
            continue

        ps_raster = rasterio.open(ps_fp)
        ps_array = ps_raster.read(window=window)
        ps_array = ps_array.astype(float)

        # read s2
        s2_raster = rasterio.open(s2_fp)
        s2_window = rasterio.windows.from_bounds(*rasterio.windows.bounds(window, ps_raster.transform), transform=s2_raster.transform)
        s2_array = s2_raster.read(window=s2_window)
        s2_array = s2_array.astype(float)

        # read s1
        s1_raster = rasterio.open(s1_fp)
        s1_window = rasterio.windows.from_bounds(*rasterio.windows.bounds(window, ps_raster.transform), transform=s1_raster.transform)
        s1_array = s1_raster.read(window=s1_window)
        s1_array = s1_array.astype(float)

        # upsample s2 and s1 to ps resolution
        s2_array = torch.nn.functional.interpolate(torch.from_numpy(s2_array).unsqueeze(0), size=(ps_array.shape[1], ps_array.shape[2]), mode='bilinear', align_corners=False)[0]
        s1_array = torch.nn.functional.interpolate(torch.from_numpy(s1_array).unsqueeze(0), size=(ps_array.shape[1], ps_array.shape[2]), mode='bilinear', align_corners=False)[0]

        data = torch.cat([torch.from_numpy(ps_array), s2_array, s1_array], dim=0)[None, ...]

        # make tensor dimensions multiples of 2
        x_pad, y_pad = 0, 0
        if data.shape[2] % 2 != 0:
            data = torch.nn.functional.pad(data, (0, 0, 0, 1), mode='constant', value=0)
            x_pad = 1
        if data.shape[3] % 2 != 0:
            data = torch.nn.functional.pad(data, (0, 1, 0, 0), mode='constant', value=0)
            y_pad = 1

        patches, t_size, p0, p1, offsets = split_tensor(data, patch_size=cfg.imsize, overlap=0)

        # merge and create dataset
        dataset = torch.utils.data.TensorDataset(patches, offsets)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=cfg.num_workers)

        with torch.no_grad():
            # Run predictions for tiles and accumulate them
            counts = []
            for batch in loader:
                patch, offset = batch
                transformed = normalizer(patch).float().to(DEVICE)
                patch_counts = model(transformed)
                counts.append(patch_counts.cpu())

        counts = torch.cat(counts, dim=0)

        # reshape
        width = np.ceil((t_size[1] + p1) / cfg.imsize).astype(int)
        height = np.ceil((t_size[0] + p0) / cfg.imsize).astype(int)
        counts = counts.reshape(height, width)
        # save
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "height": counts.shape[0],
            "width": counts.shape[1],
            "transform": rasterio.transform.from_bounds(*rasterio.windows.bounds(window, ps_raster.transform), counts.shape[1], counts.shape[0]),
            "crs": ps_raster.crs,
        }
        with rasterio.open(out_fp, "w", **profile) as dst:
            dst.write(counts.numpy(), 1)
        print(counts.shape)


if __name__ == "__main__":
    test()