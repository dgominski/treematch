import torch
import os
import glob
import albumentations as A
import rasterio
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import torchvision.transforms as T
import geopandas as gpd
from skimage.feature import peak_local_max
from shapely.geometry import box
import rasterio.features
from torch.utils.data import SubsetRandomSampler, Subset, DataLoader

mean = [17.85, 26.00, 26.36, 81.41]
std = [10.08, 9.94, 9.09, 25.76]


def load_worker(spot_fp, plot_center, points):
    spot = rasterio.open(spot_fp)
    spot_image = spot.read().astype(np.float32)[:4, :, :]  # first 4 bands
    # rasterize points
    target = np.zeros((1, spot_image.shape[1], spot_image.shape[2]), dtype=np.float32)
    if len(points) != 0:
        xs = points.geometry.x.values
        ys = points.geometry.y.values
        rows, cols = spot.index(xs, ys)
        rows = np.array(rows)
        cols = np.array(cols)
        mask = (rows >= 0) & (rows < spot_image.shape[1]) & (cols >= 0) & (cols < spot_image.shape[2])
        rows = rows[mask].astype(np.int32)
        cols = cols[mask].astype(np.int32)
        target[0, rows, cols] = 1.0

    disk = plot_center.buffer(15)
    valid = rasterio.features.rasterize(
        [(disk, 1)],
        out_shape=(spot_image.shape[1], spot_image.shape[2]),
        transform=spot.transform,
        fill=0,
        dtype=rasterio.uint8
    )

    spot.close()

    return {
        'spot': torch.from_numpy(spot_image),
        'labels': torch.from_numpy(target).bool(),
        'valid': torch.from_numpy(valid[np.newaxis, :, :]).bool()
    }


class SPOTStrong(torch.utils.data.Dataset):
    def __init__(self, imsize, split, root, preload=False, **kwargs):
        assert split in ["train_strong", "test"], "Invalid split"
        self.split = split
        self.split_dir = os.path.join(root, split)

        self.tifs = sorted(glob.glob(os.path.join(self.split_dir, "*.tif")))
        points_gdf = gpd.read_file(os.path.join(self.split_dir, "points.gpkg"), engine="pyogrio")
        self.points_by_tile = {name: g for name, g in points_gdf.groupby("tile")}

        self.crop = A.Compose([
            A.PadIfNeeded(min_height=imsize, min_width=imsize, border_mode=0, fill=0),
            A.CenterCrop(height=imsize, width=imsize),
        ],
            keypoint_params=A.KeypointParams(format='yx', remove_invisible=True),
            seed=42
        )
        self.transform = T.Compose([
            T.Normalize(mean=mean, std=std)
        ])

        self.preloaded = False
        if preload:
            self.preload()

        self.nbands = 4

    def _load_tile(self, idx):
        tif_path = self.tifs[idx]
        tile_name = os.path.splitext(os.path.basename(tif_path))[0]
        with rasterio.open(tif_path) as src:
            data = src.read()  # (5, H, W): 4 image bands + 1 validity
            tile_transform = src.transform
        im = data[:4]
        valid = data[4]
        pts_gdf = self.points_by_tile.get(tile_name)
        if pts_gdf is not None and len(pts_gdf) > 0:
            xs = pts_gdf.geometry.x.values
            ys = pts_gdf.geometry.y.values
            rows, cols = rasterio.transform.rowcol(tile_transform, xs, ys)
            points = list(zip(rows, cols))
        else:
            points = []
        return im, valid, points

    def __getitem__(self, index):
        if self.preloaded:
            im, valid, points = self.data[index]
        else:
            im, valid, points = self._load_tile(index)

        augmented = self.crop(image=np.transpose(im, (1, 2, 0)),
                              keypoints=np.array(points),
                              mask=valid)
        image = np.transpose(augmented['image'], (2, 0, 1))
        valid = augmented["mask"]
        points = augmented['keypoints']
        image = self.transform(torch.tensor(image, dtype=torch.float32))
        image = torch.cat([image, torch.from_numpy(valid[None, ])], dim=0)
        cm = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)
        points = np.array([[int(p[0]), int(p[1])] for p in points if 0 <= p[0] < image.shape[2] and 0 <= p[1] < image.shape[1]])
        if len(points) > 0:
            np.add.at(cm, (points[:, 0], points[:, 1]), 1.0)
        return image, torch.from_numpy(valid)[None,], torch.from_numpy(cm[None, :, :])

    def __len__(self):
        return len(self.tifs)

    def preload(self):
        self.data = []
        for idx in range(len(self.tifs)):
            self.data.append(self._load_tile(idx))
        self.preloaded = True

    def to_disk(self):
        self.plots = gpd.read_file("/data/Open-Canopy/datasets/count/plots.gpkg")
        self.points = gpd.read_file("/data/Open-Canopy/datasets/count/points.gpkg")
        self.points = self.points[self.points["geometry"].is_valid].reset_index(drop=True)
        self.plots = gpd.read_file(f"/data/Open-Canopy/datasets/count/{self.split}_plots.gpkg")
        out_dir = f"/data/Open-Canopy/datasets/count/pt/{self.split}/"
        # save as .pt
        for i in tqdm(range(len(self.plots)), desc="Saving dataset to disk"):
            plot = self.plots.iloc[i]
            plot_id = plot['plot_id']
            spot_fp = f"/data/Open-Canopy/datasets/count/plots/plot_{plot_id}.tif"
            points = self.points[self.points['plot_id'] == plot_id]
            d = load_worker(spot_fp, plot.geometry, points)
            out_fp = os.path.join(out_dir, f"plot_{plot_id}.pt")
            torch.save(d, out_fp)


class SPOTWeak(torch.utils.data.Dataset):
    def __init__(self, imsize, root, imagery_root, preload=False, **kwargs):
        self.imagery_root = imagery_root

        self.geometries = gpd.read_file(os.path.join(root, "geometries.geojson"))
        pseudolabel_dir = os.path.join(root, "pseudolabels")
        self.pseudolabel_dir = pseudolabel_dir
        self.point_fps = glob.glob(os.path.join(pseudolabel_dir, "*.gpkg"))
        on_disk = set([os.path.basename(fp).split("cropped_")[1].split(".gpkg")[0] for fp in self.point_fps])
        self.geometries = self.geometries[self.geometries['crop_id'].astype(str).isin(on_disk)].reset_index(drop=True)

        band_stats = np.load("data/spot_band_stats.npz")
        self.crop = A.Compose([
            A.PadIfNeeded(min_height=imsize, min_width=imsize, border_mode=0, fill=0),
            A.CenterCrop(height=imsize, width=imsize),
        ],
            additional_targets={'valid': 'mask'}
        )
        self.transform = T.Compose([
            T.Normalize(mean=band_stats['mean'].tolist(), std=band_stats['std'].tolist())
        ])
        self.preloaded = False
        self.nbands = 4

        self.imsize = imsize

    def __getitem__(self, index):
        geom = self.geometries.iloc[index]
        tile_id = geom['image_name'].split("compressed_pansharpened_")[1].split(".tif")[0]
        lidar_crop_id = geom["crop_id"]
        year = geom['lidar_year']
        spot_fp = os.path.join(self.imagery_root, str(year), "spot", f"compressed_pansharpened_{tile_id}.tif")
        point_fp = os.path.join(self.pseudolabel_dir, f"cropped_{lidar_crop_id}.gpkg")

        spot = rasterio.open(spot_fp)
        # get geom bounds in pixel coordinates
        geom_bounds = geom.geometry.bounds
        min_row, min_col = spot.index(geom_bounds[0], geom_bounds[3])  # minx, maxy
        max_row, max_col = spot.index(geom_bounds[2], geom_bounds[1])
        # sample a random imsize x imsize crop within the geometry, pixel coordinates
        x = np.random.randint(min_col, max_col - self.imsize)
        y = np.random.randint(min_row, max_row - self.imsize)
        spot_window = rasterio.windows.Window(x, y, self.imsize, self.imsize)
        spot_crop = spot.read(window=spot_window).astype(np.float32)
        window_transform = spot.window_transform(spot_window)

        #convert window to real world coordinates
        x_min, y_min = rasterio.transform.xy(window_transform, 0, 0, offset='ul')
        x_max, y_max = rasterio.transform.xy(window_transform, self.imsize, self.imsize, offset='lr')
        points_gdf = gpd.read_file(point_fp, bbox=(x_min, y_min, x_max, y_max))
        xs = points_gdf.geometry.x.values
        ys = points_gdf.geometry.y.values
        rows, cols = spot.index(xs, ys)
        spot.close()

        target = np.zeros((1, spot_crop.shape[1], spot_crop.shape[2]), dtype=np.float32)
        coords = np.column_stack((rows, cols)) - np.array([[y, x]])
        mask = (coords[:, 0] >= 0) & (coords[:, 0] < spot_crop.shape[1]) & (coords[:, 1] >= 0) & (coords[:, 1] < spot_crop.shape[2])
        rows = coords[mask, 0].astype(np.int32)
        cols = coords[mask, 1].astype(np.int32)
        target[0, rows, cols] = 1.0

        inp = spot_crop
        valid = np.ones((1, inp.shape[1], inp.shape[2]), dtype=np.float32)
        # apply random crop
        augmented = self.crop(image=inp.transpose(1, 2, 0),
                              mask=target.transpose(1, 2, 0).astype(np.uint8),
                              valid=valid.transpose(1, 2, 0).astype(np.uint8))
        raw_image = np.transpose(augmented['image'], (2, 0, 1))
        image = self.transform(torch.tensor(raw_image, dtype=torch.float32))
        valid = torch.from_numpy(augmented['valid'].transpose(2, 0, 1)).float()
        target = torch.from_numpy(augmented['mask'].transpose(2, 0, 1)).float()

        inp = torch.cat([image, valid], dim=0)  # append valid mask as last channel
        return inp, valid, target

    def __len__(self):
        return len(self.geometries)

    def loader(self, batch_size, num_workers):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


