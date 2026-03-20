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


points_gdf_fp = "/scratch/gf2/points_trees_GF_2_China.gpkg"

mean = [73.2, 80.2, 72.7, 105.7]
std = [39.1, 39.1, 41.4, 53.5]


class GFCountingDataset(torch.utils.data.Dataset):
    def __init__(self, imsize, split, preload=True, **kwargs):
        assert split in ["train", "test"], "Invalid split"
        self.split = split
        self.imsize = imsize

        pt_dir = f"/scratch/gf2/pt/{split}/"
        self.pts = glob.glob(os.path.join(pt_dir, "*.pt"))

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

    def __len__(self):
        return len(self.pts)

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
        image = self.transform(torch.tensor(image, dtype=torch.float32))
        image = torch.cat([image, torch.ones(1, image.shape[1], image.shape[2])], dim=0)
        # convert point list to count map vectorized
        cm = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)
        points = np.array([[int(p[0]), int(p[1])] for p in points if 0 <= p[0] < image.shape[2] and 0 <= p[1] < image.shape[1]])
        if len(points) > 0:
            cm[points[:, 0], points[:, 1]] = 1.0
        return image, torch.from_numpy(cm[None, :, :])

    def to_disk(self, split):
        out_dir = f"/scratch/gf2/pt/{split}/"
        os.makedirs(out_dir, exist_ok=True)

        im_root = "/scratch/gf2/GF2_images_China"
        nbands = 4

        points_gdf = gpd.read_file(points_gdf_fp).to_crs("EPSG:4326")
        rectangle_fp = "/scratch/gf2/train_rectangles.gpkg" if split == "train" else "/scratch/gf2/test_tiles.gpkg"
        rectangles = gpd.read_file(rectangle_fp).to_crs("EPSG:4326").reset_index(drop=True)
        # remove rectangles with no points
        joined = gpd.sjoin(
            rectangles,
            points_gdf,
            how="inner",
            predicate="contains"
        )
        rectangles = rectangles.iloc[joined.index.unique().tolist()]
        # spatial join to get im_id for each point
        rectangles = rectangles.reset_index(drop=True)
        rectangles["rec_id"] = rectangles.index
        points_gdf = gpd.sjoin(
            points_gdf,
            rectangles[['geometry', 'rec_id']],
            how='inner',
            predicate='intersects'
        )
        points_gdf = points_gdf.drop(columns=['index_right'])

        for idx in range(len(rectangles)):
            row = rectangles.iloc[idx]
            file_idx = row.id
            rectangle = row.geometry
            im_fp = f"{im_root}/rectangle_{file_idx}.tif"
            points = points_gdf[points_gdf['rec_id'] == idx]
            with rasterio.open(im_fp) as src:
                # convert rectangle to window
                window = rasterio.windows.from_bounds(
                    *rectangle.bounds,
                    transform=src.transform
                )
                window_transform = src.window_transform(window)
                im = src.read(
                    indexes=list(range(1, nbands + 1)),
                    window=window,
                    boundless=True,
                    fill_value=0
                )
                # project points to pixel coordinates
                xs = points.geometry.x.values
                ys = points.geometry.y.values
                rows, cols = rasterio.transform.rowcol(window_transform, xs, ys)

            points = np.column_stack((rows, cols))
            mask = (points[:, 1] >= 0) & (points[:, 1] < im.shape[2]) & (points[:, 0] >= 0) & (
                        points[:, 0] < im.shape[1])
            points = points[mask]

            points = torch.tensor(points, dtype=torch.int64)
            im = torch.tensor(im, dtype=torch.float32)
            # save
            torch.save({
                "im": im,
                "points": points
            }, os.path.join(out_dir, f"sample_{idx}.pt"))

    def preload(self):
        self.data = []
        for idx in range(len(self.pts)):
            data = torch.load(self.pts[idx])
            self.data.append(data)
        self.preloaded = True


class GFPseudoLabelDataset(torch.utils.data.Dataset):
    def __init__(self, imsize, preload=False, **kwargs):
        self.root = "/scratch/gf2/unlabeled/"
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


def make_split():
    rectangles_gdf = gpd.read_file(rectangles_gdf_fp)
    rectangles_gdf["id"] = rectangles_gdf.index.astype(int)
    chm_bounds = gpd.read_file("/scratch/gf2/chm_bounds.gpkg")
    test_area = gpd.read_file("/scratch/gf2/test_area.gpkg")
    # isolate test rectangles
    test_rects = gpd.sjoin(rectangles_gdf, test_area[['geometry']], how='inner', predicate='intersects')
    train_rects = rectangles_gdf[~rectangles_gdf.index.isin(test_rects.index)]
    train_rects.to_file("/scratch/gf2/train_rectangles.gpkg", driver="GPKG")
    test_rects.to_file("/scratch/gf2/test_rectangles.gpkg", driver="GPKG")
    # select 20 chms that intersect train rectangles
    train_chm_bounds = gpd.sjoin(chm_bounds, train_rects[['geometry']], how='inner', predicate='intersects')
    train_chm_bounds = train_chm_bounds.drop_duplicates(subset=['fp'])
    selected_chms = train_chm_bounds.sample(n=20, random_state=42)
    selected_chms.to_file("/scratch/gf2/selected_train_chm_bounds.gpkg", driver="GPKG")

    train_rects = gpd.read_file("/scratch/gf2/train_rectangles.gpkg").to_crs("EPSG:3857")
    test_rects = gpd.read_file("/scratch/gf2/test_rectangles.gpkg").to_crs("EPSG:3857")
    points = gpd.read_file(points_gdf_fp).to_crs("EPSG:3857")
    train_area = train_rects.geometry.area.sum() / 1e6
    test_area = test_rects.geometry.area.sum() / 1e6
    train_points = gpd.sjoin(points, train_rects[['geometry']], how='inner', predicate='within')
    test_points = gpd.sjoin(points, test_rects[['geometry']], how='inner', predicate='within')
    print(f"Train area (km^2): {train_area}, Number of train points: {len(train_points)}")
    print(f"Test area (km^2): {test_area}, Number of test points: {len(test_points)}")


def fetch_chms():
    import shutil
    chm_gdf = gpd.read_file("/scratch/gf2/selected_train_chm_bounds.gpkg")
    in_im_dir = "/bigdata/gaofen/GF2/China/gf2/gf2_images_v3"
    out_im_dir = "/scratch/gf2/unlabeled/ims/"
    out_chm_dir = "/scratch/gf2/unlabeled/chm/"
    for fp in chm_gdf['fp']:
        im_fp = os.path.join(in_im_dir, os.path.basename(fp).replace("_scaled_int.tif", ".jp2"))
        shutil.copy(fp, out_chm_dir)
        shutil.copy(im_fp, out_im_dir)


def generate_test_windows():
    from rasterio.windows import Window
    rectangles = gpd.read_file("/scratch/gf2/test_rectangles.gpkg").to_crs("EPSG:4326")
    tiles = []
    rect_ids = []
    tile_height = 64
    tile_width = 64
    for row in rectangles.itertuples():
        file_idx = row.id
        geom = row.geometry
        raster_path = f"/scratch/gf2/GF2_images_China/rectangle_{file_idx}.tif"

        with rasterio.open(raster_path) as src:
            transform = src.transform
            crs = src.crs

            # get bounding box in world coords
            xmin, ymin, xmax, ymax = geom.bounds

            # convert bounds to pixel coords
            row_min, col_min = rasterio.transform.rowcol(transform, xmin, ymax)
            row_max, col_max = rasterio.transform.rowcol(transform, xmax, ymin)

            # ensure ordering
            r0, r1 = sorted([row_min, row_max])
            c0, c1 = sorted([col_min, col_max])

            height = r1 - r0
            width = c1 - c0

            n_rows = int(np.ceil(height / tile_height))
            n_cols = int(np.ceil(width / tile_width))

            row_starts = [
                r0 + i * tile_height for i in range(n_rows)
            ]
            col_starts = [
                c0 + j * tile_width for j in range(n_cols)
            ]

            # force last tiles to align with end → overlap if needed
            row_starts[-1] = max(r1 - tile_height, r0)
            col_starts[-1] = max(c1 - tile_width, c0)

            for r in row_starts:
                for c in col_starts:
                    win = Window(
                        col_off=c,
                        row_off=r,
                        width=tile_width,
                        height=tile_height
                    )

                    w_tr = src.window_transform(win)
                    x0, y0 = w_tr * (0, 0)
                    x1, y1 = w_tr * (tile_width, tile_height)

                    tiles.append(box(x0, y1, x1, y0))
                    rect_ids.append(file_idx)

    tiles_gdf = gpd.GeoDataFrame(geometry=tiles, crs=crs)
    tiles_gdf["id"] = rect_ids
    tiles_gdf.to_file("/scratch/gf2/test_tiles.gpkg", driver="GPKG")


if __name__ == "__main__":
    from itertools import cycle
    import matplotlib.pyplot as plt
    from torch.utils.data import random_split
    from tqdm import tqdm
    from omegaconf import OmegaConf

    #print total rectangle area in meters and number of points
    # rectangles_gdf = gpd.read_file(rectangles_gdf_fp).to_crs("EPSG:4545")
    # points_gdf = gpd.read_file(points_gdf_fp).to_crs("EPSG:4545")
    # print(f"Total rectangle area (km^2): {rectangles_gdf.geometry.area.sum() / 1e6}, Number of points: {len(points_gdf)}")
    # exit()
    # cfg = OmegaConf.create({
    #         "imsize": 512,
    #     })

    # point_dir = "/scratch/gf2/unlabeled/points/"
    # total_area = 0.0
    # nb_trees = 0
    # for point_fp in tqdm(glob.glob(os.path.join(point_dir, "*.geojson"))):
    #     gdf = gpd.read_file(point_fp).to_crs("EPSG:3857")
    #     # isolate rectangle and points
    #     rect = shape(gdf.geometry.iloc[0])
    #     points = gdf.geometry.iloc[1:]
    #     total_area += rect.area / 1e6
    #     nb_trees += len(points)
    # print(f"Total unlabeled area (km^2): {total_area}, Number of unlabeled points: {nb_trees}")
    # exit()

    # #compute average number of tree per pixel in labeled train set
    # train_dataset = GFCountingDataset(imsize=64, split="train", preload=True)
    # nb_trees = 0
    # total_pixels = 0
    # for im, cm in tqdm(train_dataset):
    #     nb_trees += cm.sum().item()
    #     total_pixels += im.shape[1] * im.shape[2]
    # avg_trees_per_pixel = nb_trees / total_pixels
    # print(f"Average number of trees per pixel in train set: {avg_trees_per_pixel}")
    # exit()
    # instantiate dataset
    train_dataset = GFPseudoLabelDataset(imsize=256, split="train")

    for i in range(50):
        im, cm = train_dataset[np.random.randint(0, len(train_dataset))]
        unnormed = im[:4] * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(torch.permute(unnormed[:3, :, :].long(), (1, 2, 0)).numpy())
        axes[0].set_title("Image")
        axes[1].imshow(cm[0, :, :], cmap='hot')
        axes[1].set_title("Count Map")
        plt.show()

    # test_dataset = GFCountingDataset(imsize=64, split="test")
    # #
    # val_fraction = 0.2
    # n_val = int(len(train_dataset) * val_fraction)
    # n_train = len(train_dataset) - n_val
    # train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val])
    # #
    # batch_size = 16
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # # get count distributions
    # train_counts = []
    # for ims, cms in tqdm(train_loader):
    #     counts = cms.view(cms.shape[0], -1).sum(dim=1)
    #     train_counts.extend(counts.numpy().tolist())
    # val_counts = []
    # for ims, cms in tqdm(val_loader):
    #     counts = cms.view(cms.shape[0], -1).sum(dim=1)
    #     val_counts.extend(counts.numpy().tolist())
    # test_counts = []
    # for ims, cms in tqdm(test_loader):
    #     counts = cms.view(cms.shape[0], -1).sum(dim=1)
    #     test_counts.extend(counts.numpy().tolist())
    # # print(train_counts, val_counts, test_counts)
    # fig, axes = plt.subplots(1, 3, figsize=(8, 6))
    # axes[0].hist(train_counts, bins=30, color='blue', alpha=0.7, label='Train')
    # axes[0].set_title("Train Set Tree Count Distribution")
    # axes[1].hist(val_counts, bins=30, color='orange', alpha=0.7, label='Validation')
    # axes[1].set_title("Validation Set Tree Count Distribution")
    # print(test_counts)
    # axes[2].hist(test_counts, bins=30, color='green', alpha=0.7, label='Test')
    # axes[2].set_title("Test Set Tree Count Distribution")
    # plt.show()