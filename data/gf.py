import glob
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import rasterio
from shapely.geometry import shape, Polygon, MultiPolygon, box
from rasterio.features import shapes
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import albumentations as A
import random
from skimage.feature import peak_local_max
from torch.utils.data import SubsetRandomSampler, Subset, DataLoader
from scipy.ndimage import gaussian_filter


mean = [73.2, 80.2, 72.7, 105.7]
std = [39.1, 39.1, 41.4, 53.5]


class GaofenStrong(torch.utils.data.Dataset):
    def __init__(self, imsize, split, root, preload=True, **kwargs):
        assert split in ["train", "test"], "Invalid split"
        self.split = split
        self.imsize = imsize

        self.pts = glob.glob(os.path.join(root, split, "*.pt"))

        self.preloaded = False
        if preload:
            self.preload()

        self.transform = T.Compose([
            T.Normalize(mean=mean, std=std)
        ])
        self.crop = A.Compose([
            A.PadIfNeeded(min_height=imsize, min_width=imsize, border_mode=0, fill=0),
            A.RandomCrop(height=imsize, width=imsize) if split == "train" else A.CenterCrop(height=imsize, width=imsize)
        ],
            keypoint_params=A.KeypointParams(format='yx', remove_invisible=True),
            seed=42
        )
        self.nbands = 4

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        if self.preloaded:
            data = self.data[idx]
        else:
            data = torch.load(self.pts[idx])
        im = data["im"].numpy()
        points = data["points"].numpy().tolist()
        valid = data["valid"].numpy()

        # apply random crop
        augmented = self.crop(image=np.transpose(im, (1, 2, 0)),
                                keypoints=np.array(points),
                              mask=valid
                              )
        image = np.transpose(augmented['image'], (2, 0, 1))
        valid = augmented["mask"]
        points = augmented['keypoints']
        image = self.transform(torch.tensor(image, dtype=torch.float32))
        image = torch.cat([image, torch.from_numpy(valid[None, ])], dim=0)
        # convert point list to count map vectorized
        cm = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)
        points = np.array([[int(p[0]), int(p[1])] for p in points if 0 <= p[0] < image.shape[2] and 0 <= p[1] < image.shape[1]])
        if len(points) > 0:
            np.add.at(cm, (points[:, 0], points[:, 1]), 1.0)
        return image, torch.from_numpy(valid), torch.from_numpy(cm[None, :, :])

    def preload(self):
        self.data = []
        for idx in range(len(self.pts)):
            data = torch.load(self.pts[idx])
            self.data.append(data)
        self.preloaded = True


class GaofenWeak(torch.utils.data.Dataset):
    def __init__(self, imsize, root, **kwargs):
        self.root = root
        self.random_crop = A.Compose([
            A.PadIfNeeded(min_height=imsize, min_width=imsize, border_mode=0, fill=0),
            A.RandomCrop(height=imsize, width=imsize)
        ],
            additional_targets={'chm': 'mask'},
            keypoint_params=A.KeypointParams(format='yx', remove_invisible=True))
        self.normalize = T.Compose([
            T.Normalize(mean=mean, std=std)
        ])
        self.nbands = 4
        self.imsize = imsize

        self.im_fps = glob.glob(os.path.join(self.root, "ims", "*.jp2"))
        self.point_fps = glob.glob(os.path.join(self.root, "points", "*.geojson"))

    def __len__(self):
        return len(self.point_fps)

    def __getitem__(self, idx):
        point_fp = self.point_fps[idx]
        im_fp = os.path.join(self.root, "ims", os.path.basename(point_fp).split("_scaled_int_")[0] + ".jp2")
        chm_fp = os.path.join(self.root, "chm", os.path.basename(point_fp).split("_scaled_int_")[0] + "_scaled_int.tif")

        gdf = gpd.read_file(point_fp)
        rectangle = gdf.geometry.iloc[-1]
        points = gdf.geometry.iloc[:-1].to_frame()

        with rasterio.open(im_fp) as src:
            # convert rectangle to window
            window = rasterio.windows.from_bounds(
                *rectangle.bounds,
                transform=src.transform
            )
            window_transform = src.window_transform(window)
            im = src.read(
                indexes=list(range(1, self.nbands + 1)),
                window=window,
                boundless=True,
                fill_value=0
            )
            # project points to pixel coordinates
            xs = points.geometry.x.values
            ys = points.geometry.y.values
            rows, cols = rasterio.transform.rowcol(window_transform, xs, ys)

        with rasterio.open(chm_fp) as src:
            chm = src.read(
                indexes=1,
                window=window,
                boundless=True,
                fill_value=0
            ) / 100.0  # scale heights

        points = np.column_stack((rows, cols))
        mask = (points[:, 1] >= 0) & (points[:, 1] < im.shape[2]) & (points[:, 0] >= 0) & (points[:, 0] < im.shape[1])
        points = points[mask]
        crop = self.random_crop(image=np.transpose(im, (1, 2, 0)), keypoints=points, chm=chm)
        im = np.transpose(crop['image'], (2, 0, 1))
        points = crop['keypoints']
        cm = np.zeros((self.imsize, self.imsize), dtype=np.float32)
        for coord in points:
            row, col = int(coord[0]), int(coord[1])
            cm[row, col] += 1.0

        im = torch.from_numpy(im).float()
        im = self.normalize(im)
        im = torch.cat([im, torch.ones(1, im.shape[1], im.shape[2])], dim=0)  # add valid mask

        chm = crop['chm']
        # pseudo_density = self.chm_to_density(chm)
        # return im, pseudo_density[None, :, :]
        return im, torch.from_numpy(cm[None, :, :])

    def loader(self, batch_size, num_workers):
        return DataLoader(self, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    @staticmethod
    def chm_to_density(chm, h_min=3.0, avg_trees_per_pixel=0.006):
        mask = (chm > h_min)
        dens = mask / (mask.sum() + 1e-6)
        dens = dens * (avg_trees_per_pixel * chm.size)
        return dens


