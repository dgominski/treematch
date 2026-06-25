import torch
import os
import glob
import albumentations as A
import rasterio
import numpy as np
import geopandas as gpd
import torchvision.transforms as T

mean = [420, 600, 640, 2100]
std = [250, 340, 415, 1170]


class PlanetScopeStrong(torch.utils.data.Dataset):
    def __init__(self, imsize, split, root, preload=False, **kwargs):
        assert split in ["train_strong", "test"], "Invalid split"
        self.split = split
        self.imsize = imsize
        self.split_dir = os.path.join(root, split)

        self.tifs = sorted(glob.glob(os.path.join(self.split_dir, "*.tif")))
        points_gdf = gpd.read_file(os.path.join(self.split_dir, "points.gpkg"), engine="pyogrio")
        self.points_by_tile = {name: g for name, g in points_gdf.groupby("tile")}

        self.preloaded = False
        if preload:
            self.preload()

        self.transform = T.Compose([
            T.Normalize(mean=mean, std=std)
        ])
        self.crop = A.Compose([
            A.PadIfNeeded(min_height=imsize, min_width=imsize, border_mode=0, fill=0),
            A.RandomCrop(height=imsize, width=imsize) if split == "train_strong" else A.CenterCrop(height=imsize, width=imsize)
        ],
            keypoint_params=A.KeypointParams(format='yx', remove_invisible=True),
            seed=42
        )
        self.nbands = 4

    def _load_tile(self, idx):
        tif_path = self.tifs[idx]
        tile_name = os.path.splitext(os.path.basename(tif_path))[0]
        with rasterio.open(tif_path) as src:
            data = src.read()
            tile_transform = src.transform
        im = data[:4]
        valid = data[4].astype(np.uint8)
        pts_gdf = self.points_by_tile.get(tile_name)
        if pts_gdf is not None and len(pts_gdf) > 0:
            xs = pts_gdf.geometry.x.values
            ys = pts_gdf.geometry.y.values
            rows, cols = rasterio.transform.rowcol(tile_transform, xs, ys)
            points = list(zip(rows, cols))
        else:
            points = []
        return im, valid, points

    def __getitem__(self, idx):
        if self.preloaded:
            im, valid, points = self.data[idx]
        else:
            im, valid, points = self._load_tile(idx)

        augmented = self.crop(image=np.transpose(im, (1, 2, 0)),
                              keypoints=np.array(points),
                              mask=valid)
        image = np.transpose(augmented['image'], (2, 0, 1))
        valid = augmented["mask"]
        points = augmented['keypoints']
        image = self.transform(torch.from_numpy(image.astype(np.float32)))
        image = torch.cat([image, torch.from_numpy(valid[None, ].astype(np.float32))], dim=0)
        cm = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)
        points = np.array([[int(p[0]), int(p[1])] for p in points if 0 <= p[0] < image.shape[2] and 0 <= p[1] < image.shape[1]])
        if len(points) > 0:
            cm[points[:, 0], points[:, 1]] = 1.0
        return image, torch.from_numpy(valid)[None], torch.from_numpy(cm[None, :, :])

    def __len__(self):
        return len(self.tifs)

    def preload(self):
        self.data = []
        for idx in range(len(self.tifs)):
            self.data.append(self._load_tile(idx))
        self.preloaded = True


class PlanetScopeWeak(torch.utils.data.Dataset):
    def __init__(self, imsize, root, preload=False, **kwargs):
        self.imsize = imsize

        self.tifs = sorted(glob.glob(os.path.join(root, "*.tif")))
        points_gdf = gpd.read_file(os.path.join(root, "points.gpkg"), engine="pyogrio")
        self.points_by_tile = {name: g for name, g in points_gdf.groupby("tile")}

        self.preloaded = False
        if preload:
            self.preload()

        self.transform = T.Compose([
            T.Normalize(mean=mean, std=std)
        ])
        self.crop = A.Compose([
            A.PadIfNeeded(min_height=imsize, min_width=imsize, border_mode=0, fill=0),
            A.RandomCrop(height=imsize, width=imsize)
        ],
            keypoint_params=A.KeypointParams(format='yx', remove_invisible=True),
            seed=42
        )
        self.nbands = 4

    def _load_tile(self, idx):
        tif_path = self.tifs[idx]
        tile_name = os.path.splitext(os.path.basename(tif_path))[0]
        with rasterio.open(tif_path) as src:
            data = src.read()
            tile_transform = src.transform
        im = data[:4]
        valid = data[4].astype(np.uint8)
        pts_gdf = self.points_by_tile.get(tile_name)
        if pts_gdf is not None and len(pts_gdf) > 0:
            xs = pts_gdf.geometry.x.values
            ys = pts_gdf.geometry.y.values
            rows, cols = rasterio.transform.rowcol(tile_transform, xs, ys)
            points = list(zip(rows, cols))
        else:
            points = []
        return im, valid, points

    def __getitem__(self, idx):
        if self.preloaded:
            im, valid, points = self.data[idx]
        else:
            im, valid, points = self._load_tile(idx)

        augmented = self.crop(image=np.transpose(im, (1, 2, 0)),
                              keypoints=np.array(points),
                              mask=valid)
        image = np.transpose(augmented['image'], (2, 0, 1))
        points = augmented['keypoints']
        valid = augmented["mask"]
        image = self.transform(torch.from_numpy(image.astype(np.float32)))
        image = torch.cat([image, torch.from_numpy(valid[None,].astype(np.float32))], dim=0)
        cm = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)
        points = np.array(
            [[int(p[0]), int(p[1])] for p in points if 0 <= p[0] < image.shape[2] and 0 <= p[1] < image.shape[1]])
        if len(points) > 0:
            np.add.at(cm, (points[:, 0], points[:, 1]), 1.0)
        return image, torch.from_numpy(valid)[None,], torch.from_numpy(cm[None, :, :])

    def __len__(self):
        return len(self.tifs)

    def loader(self, batch_size, num_workers):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    def preload(self):
        self.data = []
        for idx in range(len(self.tifs)):
            self.data.append(self._load_tile(idx))
        self.preloaded = True
