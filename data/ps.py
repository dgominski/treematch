import torch
import os
import glob
import albumentations as A
import rasterio
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import torchvision.transforms as T
from collections import defaultdict
import rasterio.features

mean = [420, 600, 640, 2100]
std = [250, 340, 415, 1170]


def load_worker(tile_path):
    return torch.load(tile_path)


class PlanetScopeStrong(torch.utils.data.Dataset):
    def __init__(self, imsize, split, root, preload=False, **kwargs):
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

    def __len__(self):
        return len(self.pts)

    def preload(self):
        self.data = []
        for idx in range(len(self.pts)):
            data = torch.load(self.pts[idx])
            self.data.append(data)
        self.preloaded = True


class PlanetScopeWeak(torch.utils.data.Dataset):
    def __init__(self, imsize, root, preload=False, **kwargs):
        self.imsize = imsize

        self.pts = glob.glob(os.path.join(root, "*.pt"))

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

    def __getitem__(self, idx):
        if self.preloaded:
            data = self.data[idx]
        else:
            data = torch.load(self.pts[idx])
        im = data["im"].numpy()
        points = data["points"].numpy().tolist()

        # apply random crop
        augmented = self.crop(image=np.transpose(im, (1, 2, 0)),
                              keypoints=np.array(points))
        image = np.transpose(augmented['image'], (2, 0, 1))
        points = augmented['keypoints']
        valid = image.sum(0) > 0
        image = self.transform(torch.tensor(image, dtype=torch.float32))
        image = torch.cat([image, torch.from_numpy(valid[None,])], dim=0)
        # convert point list to count map vectorized
        cm = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)
        points = np.array(
            [[int(p[0]), int(p[1])] for p in points if 0 <= p[0] < image.shape[2] and 0 <= p[1] < image.shape[1]])
        if len(points) > 0:
            np.add.at(cm, (points[:, 0], points[:, 1]), 1.0)
        return image, torch.from_numpy(valid), cm[None, :, :]

    def __len__(self):
        return len(self.pts)

    def loader(self, batch_size, num_workers):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    def preload(self):
        self.data = []
        for idx in range(len(self.pts)):
            data = torch.load(self.pts[idx])
            self.data.append(data)
        self.preloaded = True
