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


class PSCountingDataset(torch.utils.data.Dataset):
    def __init__(self, imsize, split, preload=False, **kwargs):
        assert split in ["train", "test"], "Invalid split"
        self.split = split
        self.imsize = imsize

        pt_dir = f"/scratch/maurice_counting/pt/{split}/"
        # pt_dir = f"/scratch/maurice_counting/pt_128/{split}/"
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

    def __getitem__(self, idx):
        if self.preloaded:
            data = self.data[idx]
        else:
            data = torch.load(self.pts[idx])
        im = data["im"].numpy()
        points = data["points"].numpy().tolist()
        # valid = data["valid"].numpy()

        # apply random crop
        augmented = self.crop(image=np.transpose(im, (1, 2, 0)),
                                keypoints=np.array(points),
                              # mask=valid
                              )
        image = np.transpose(augmented['image'], (2, 0, 1))
        # valid = augmented["mask"]
        valid = image.sum(0) > 0

        points = augmented['keypoints']
        image = self.transform(torch.tensor(image, dtype=torch.float32))
        image = torch.cat([image, torch.from_numpy(valid[None, ])], dim=0)
        # convert point list to count map vectorized
        cm = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)
        points = np.array([[int(p[0]), int(p[1])] for p in points if 0 <= p[0] < image.shape[2] and 0 <= p[1] < image.shape[1]])
        if len(points) > 0:
            cm[points[:, 0], points[:, 1]] = 1.0
        return image, torch.from_numpy(cm[None, :, :])

    def __len__(self):
        return len(self.pts)

    @staticmethod
    def to_disk(split):
        from shapely.geometry import Polygon, mapping
        out_dir = f"/scratch/maurice_counting/pt_128/{split}"
        os.makedirs(out_dir, exist_ok=True)

        nbands = 4

        labeling_rectangles_fp = "/scratch/maurice_counting/train_rectangles.gpkg" if split == "train" else "/scratch/maurice_counting/test_rectangles.gpkg"
        labeling_rectangles = gpd.read_file(labeling_rectangles_fp).to_crs("EPSG:4326").reset_index(drop=True)

        points_gdf = gpd.read_file("/scratch/maurice_counting/clean_points.gpkg").to_crs("EPSG:4326")
        rectangle_fp = "/scratch/maurice_counting/train_rectangles.gpkg" if split == "train" else "/scratch/maurice_counting/test_tiles_128.gpkg"
        rectangles = gpd.read_file(rectangle_fp).to_crs("EPSG:4326").reset_index(drop=True)
        ps_footprints = gpd.read_file("/scratch/maurice_counting/ps/ps_footprints.gpkg")
        rectangles = rectangles.sjoin(ps_footprints, predicate="within").drop(columns=["index_right"]).reset_index(drop=True)

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
            rectangle = row.geometry
            im_fp = row.fp
            # points = points_gdf[points_gdf['rec_id'] == idx]
            with rasterio.open(im_fp) as src:
                transform = src.transform

                # get rectangle bounds in pixel coordinates
                row_min, col_min = rasterio.transform.rowcol(transform, rectangle.bounds[0], rectangle.bounds[3])
                row_max, col_max = rasterio.transform.rowcol(transform, rectangle.bounds[2], rectangle.bounds[1])
                r0, r1 = sorted([row_min, row_max])
                c0, c1 = sorted([col_min, col_max])
                height = r1 - r0
                width = c1 - c0
                # enforce minimal window size
                extra_h = max(0, 128 - height)
                extra_w = max(0, 128 - width)
                # center the rectangle within the window
                r0 = r0 - extra_h // 2
                r1 = r1 + extra_h - extra_h // 2
                c0 = c0 - extra_w // 2
                c1 = c1 + extra_w - extra_w // 2
                # make sure window stays within raster bounds
                r0 = max(r0, 0)
                c0 = max(c0, 0)
                r1 = min(r1, src.height)
                c1 = min(c1, src.width)

                window = rasterio.windows.Window(
                    col_off=c0,
                    row_off=r0,
                    width=c1 - c0,
                    height=r1 - r0
                )

                window_transform = src.window_transform(window)
                im = src.read(
                    indexes=list(range(1, nbands + 1)),
                    window=window,
                    boundless=True,
                    fill_value=0
                )

                height = im.shape[1]
                width = im.shape[2]

                # --- rasterize validity mask ---
                valid = rasterio.features.rasterize(
                    [(geom, 1) for geom in labeling_rectangles.geometry],
                    out_shape=(height, width),
                    transform=window_transform,
                    fill=0,
                    dtype=np.uint8
                )
                # project points to pixel coordinates
                xs = points_gdf.geometry.x.values
                ys = points_gdf.geometry.y.values
                rows, cols = rasterio.transform.rowcol(window_transform, xs, ys)

            points = np.column_stack((rows, cols))
            mask = (points[:, 1] >= 0) & (points[:, 1] < im.shape[2]) & (points[:, 0] >= 0) & (
                        points[:, 0] < im.shape[1])
            points = points[mask]

            points = torch.tensor(points, dtype=torch.int64)
            im = torch.tensor(im.astype(np.int32), dtype=torch.int32)
            valid = torch.tensor(valid, dtype=torch.uint8)

            # save
            torch.save({
                "im": im,
                "points": points,
                "valid": valid
            }, os.path.join(out_dir, f"sample_{idx}.pt"))

    def preload(self):
        self.data = []
        for idx in range(len(self.pts)):
            data = torch.load(self.pts[idx])
            self.data.append(data)
        self.preloaded = True


class PSUnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, imsize, preload=False, **kwargs):
        self.imsize = imsize

        pt_dir = f"/scratch/maurice_counting/pt/unlabeled/"
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
            cm[points[:, 0], points[:, 1]] = 1.0
        return image, cm[None, :, :]

    def __len__(self):
        return len(self.pts)

    def loader(self, batch_size, num_workers):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    @staticmethod
    def to_disk():
        from rasterio.windows import from_bounds
        rectangles = gpd.read_file("/scratch/maurice_counting/noisy_rectangles.gpkg")
        ps_footprints = gpd.read_file("/scratch/maurice_counting/ps/ps_footprints.gpkg")
        rectangles = rectangles.sjoin(ps_footprints, predicate="within").drop(columns=["index_right"]).reset_index(drop=True)
        out_dir = "/scratch/maurice_counting/pt/unlabeled"
        for i, row in rectangles.iterrows():
            bounds = row.geometry.bounds  # (xmin, ymin, xmax, ymax)

            # read points in rectangle bbox
            points = gpd.read_file(
                "/scratch/maurice_counting/noisy_points.gpkg",
                bbox=bounds,
                engine="pyogrio"
            )
            with rasterio.open(row["fp"]) as src:
                # convert rectangle to window
                window = rasterio.windows.from_bounds(
                    *bounds,
                    transform=src.transform
                )
                window_transform = src.window_transform(window)
                im = src.read(
                    indexes=list(range(1, 5)),
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
            im = torch.tensor(im.astype(np.int32), dtype=torch.int32)
            print(f"saving {len(points)} points")
            # save
            torch.save({
                "im": im,
                "points": points
            }, os.path.join(out_dir, f"sample_{i}.pt"))

    def preload(self):
        self.data = []
        for idx in range(len(self.pts)):
            data = torch.load(self.pts[idx])
            self.data.append(data)
        self.preloaded = True


def clean_data(points, rectangles):
    points = points.explode(index_parts=False)[["geometry"]].reset_index(drop=True)

    rectangles = rectangles.explode(index_parts=False)[["geometry"]].reset_index(drop=True)
    rectangles["geometry"] = rectangles.geometry.buffer(0)
    rectangles = rectangles.drop_duplicates(subset="geometry")

    # self-intersection join (exclude self)
    pairs = gpd.sjoin(
        rectangles,
        rectangles,
        predicate="intersects"
    )
    pairs = pairs[pairs.index != pairs.index_right]
    # align right geometries explicitly as a GeoSeries
    right_geoms = rectangles.geometry.loc[pairs.index_right].reset_index(drop=True)
    left_geoms = pairs.geometry.reset_index(drop=True)

    # compute overlaps safely
    pairs["overlap"] = gpd.GeoSeries(
        left_geoms.values.intersection(right_geoms.values),
        index=pairs.index
    )

    # union overlaps per rectangle
    overlap_union = (
        pairs.groupby(pairs.index)["overlap"]
        .apply(lambda g: unary_union(g.values))
    )

    # subtract overlaps
    rectangles.loc[overlap_union.index, "geometry"] = (
        rectangles.loc[overlap_union.index].geometry
        .difference(overlap_union)
        .buffer(0)
    )

    # remove all points not within a rectangle
    points_in_rectangles = (
        gpd.sjoin(points, rectangles, predicate="within")
           .loc[:, points.columns]
           .drop_duplicates()
    )
    rectangles_with_points = (
        gpd.sjoin(rectangles, points_in_rectangles, predicate="contains")
        .loc[:, rectangles.columns]
        .drop_duplicates()
    )
    return points_in_rectangles, rectangles_with_points


def make_split():
    rectangles_gdf = gpd.read_file("/scratch/maurice_counting/clean_rectangles.gpkg")
    rectangles_gdf["id"] = rectangles_gdf.index.astype(int)
    test_area = gpd.read_file("/scratch/maurice_counting/test_area.gpkg")
    # isolate test rectangles
    test_rects = gpd.sjoin(rectangles_gdf, test_area[['geometry']], how='inner', predicate='intersects')
    train_rects = rectangles_gdf[~rectangles_gdf.index.isin(test_rects.index)]
    train_rects.to_file("/scratch/maurice_counting/train_rectangles.gpkg", driver="GPKG")
    test_rects.to_file("/scratch/maurice_counting/test_rectangles.gpkg", driver="GPKG")

    train_rects = train_rects.to_crs("EPSG:3857")
    test_rects = test_rects.to_crs("EPSG:3857")
    points = gpd.read_file("/scratch/maurice_counting/clean_points.gpkg").to_crs("EPSG:3857")
    train_area = train_rects.geometry.area.sum() / 1e6
    test_area = test_rects.geometry.area.sum() / 1e6
    train_points = gpd.sjoin(points, train_rects[['geometry']], how='inner', predicate='within')
    test_points = gpd.sjoin(points, test_rects[['geometry']], how='inner', predicate='within')
    print(f"Train area (km^2): {train_area}, Number of train points: {len(train_points)}")
    print(f"Test area (km^2): {test_area}, Number of test points: {len(test_points)}")


def generate_test_windows(imsize=128):
    from rasterio.windows import Window
    from shapely.geometry import box
    rectangles = gpd.read_file("/scratch/maurice_counting/test_rectangles.gpkg").to_crs("EPSG:4326")
    ps_footprints = gpd.read_file("/scratch/maurice_counting/ps/ps_footprints.gpkg")
    rectangles = rectangles.sjoin(ps_footprints, predicate="within").drop(columns=["index_right"])

    tiles = []
    for row in rectangles.itertuples():
        geom = row.geometry
        raster_path = row.fp

        with rasterio.open(raster_path) as src:
            transform = src.transform
            crs = src.crs

            # rectangle bounds
            xmin, ymin, xmax, ymax = geom.bounds

            # convert bounds to pixel coords
            row_min, col_min = rasterio.transform.rowcol(transform, xmin, ymax)
            row_max, col_max = rasterio.transform.rowcol(transform, xmax, ymin)

            r0, r1 = sorted([row_min, row_max])
            c0, c1 = sorted([col_min, col_max])

            height = r1 - r0
            width = c1 - c0

            # number of tiles along each dimension
            n_rows = int(np.ceil(height / imsize))
            n_cols = int(np.ceil(width / imsize))

            # compute start positions
            if n_rows == 1:
                # center rectangle vertically
                row_starts = [r0 + (height - imsize) // 2]  # can be negative
            else:
                row_starts = [r0 + i * imsize for i in range(n_rows)]
                row_starts[-1] = max(r1 - imsize, r0)

            if n_cols == 1:
                # center rectangle horizontally
                col_starts = [c0 + (width - imsize) // 2]
            else:
                col_starts = [c0 + j * imsize for j in range(n_cols)]
                col_starts[-1] = max(c1 - imsize, c0)

            for r in row_starts:
                for c in col_starts:
                    win = Window(
                        col_off=c,
                        row_off=r,
                        width=imsize,
                        height=imsize
                    )

                    w_tr = src.window_transform(win)
                    x0, y0 = w_tr * (0, 0)
                    x1, y1 = w_tr * (imsize, imsize)

                    tiles.append(box(x0, y1, x1, y0))

    tiles_gdf = gpd.GeoDataFrame(geometry=tiles, crs=crs)
    # tiles_gdf["id"] = rect_ids
    tiles_gdf.to_file("/scratch/maurice_counting/test_tiles_128.gpkg", driver="GPKG")


def get_ps_footprints():
    from shapely.geometry import box
    ps_dir = "/scratch/maurice_counting/ps/"
    ps_fps = glob.glob(os.path.join(ps_dir, "*.jp2"))
    ps_bounds = []
    for pfp in ps_fps:
        pr = rasterio.open(pfp)
        ps_bounds.append(pr.bounds)
    bounds = np.array(ps_bounds)
    geometries = [box(b[0], b[1], b[2], b[3]) for b in bounds]
    bounds_gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
    bounds_gdf["fp"] = ps_fps
    bounds_gdf.to_file("/scratch/maurice_counting/ps/ps_footprints.gpkg")






if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from omegaconf import OmegaConf
    import geopandas as gpd
    from shapely.ops import unary_union
    from skimage.exposure import rescale_intensity
    from torch.utils.data import DataLoader, random_split

    # get_ps_footprints()
    # make_split()
    # generate_test_windows()
    # exit()
    # dataset = PSCountingDataset(imsize=128, split="test", preload=True)
    # dataset.to_disk("test")
    # dataset = PSCountingDataset(imsize=128, split="train", preload=True)
    # dataset.to_disk("train")
    # exit()
    dataset = PSUnlabeledDataset(imsize=128)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    bigarray = []
    for d in train_loader:
        im, cm = d
        unnormed = im[0, :4] * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
        fig, axes = plt.subplots(1, 3, figsize=(8, 4))
        im_for_viz = rescale_intensity(unnormed[:3].numpy())
        axes[0].imshow(im_for_viz.transpose(1, 2, 0))
        axes[0].set_title("Image")
        axes[1].imshow(cm[0, 0, :, :], cmap='hot')
        axes[1].set_title("Count Map")
        axes[2].imshow(im[0, -1])
        plt.show()
    #
    #     bigarray.append(im[:4])
    # bigarray = torch.stack(bigarray).permute(1, 0, 2, 3).flatten(1)
    # print("mean", bigarray.mean(dim=1))
    # print("std", bigarray.std(dim=1))

    # band_stats = np.load("ps_band_stats.npz")
    # print(band_stats["mean"], band_stats["std"])
    # points = gpd.read_file("/scratch/maurice_counting/noisy_rwanda.gpkg", engine="pyogrio")
    # rectangles = gpd.read_file("/scratch/maurice_counting/noisy_rectangles_rwanda.gpkg", engine="pyogrio")
    #
    # points_clean, rectangles_clean = clean_data(points, rectangles)
    # points_clean.to_file("/scratch/maurice_counting/cleaned/noisy_points.gpkg")
    # rectangles_clean.to_file("/scratch/maurice_counting/cleaned/noisy_rectangles.gpkg")

    # points_noisy = gpd.read_file("/scratch/maurice_counting/noisy_rwanda.gpkg")
    # rectangles_noisy = gpd.read_file("/scratch/maurice_counting/noisy_rectangles_rwanda.gpkg")

    # rwanda = gpd.read_file("/scratch/maurice_counting/rwanda.gpkg")
    # exit()

    dataroot = "/scratch/maurice_counting/pt/test/"
    tiles = glob.glob(os.path.join(dataroot, "*.pt"))
    # compute total area and total tree counts
    total_area = 0.0
    total_trees = 0.0
    for tile_path in tqdm(tiles, total=len(tiles), desc="Computing total area"):
        d = torch.load(tile_path)
        im = d["im"].numpy()
        valid = im.sum(0) > 0
        pixel_area = 3 * 3  # m^2
        area = (valid.sum().item()) * pixel_area
        total_area += area
        trees = len(d["points"])
        total_trees += trees
    print(f"Total area covered by dataset: {total_area/1e6:.2f} km^2")
    print(f"Total number of trees in dataset: {total_trees:.2f}")
    exit()

    # cfg = OmegaConf.create({
    #     "imsize": 32,
    # })
    # dataset = CountingDataset(cfg)
    # dataset.to_disk()
    #
    # big_array = []
    # for i in np.random.randint(0, len(dataset), 1000):
    #     a = dataset[i]
    #     im = a[0]
    #     big_array.append(im.numpy())
    # big_array = np.stack(big_array, axis=0)
    # #print mean std for each band
    # m = big_array.mean(axis=(0, 2, 3))
    # s = big_array.std(axis=(0, 2, 3))
    # print(m, s)
    # save
    # np.savez("data/ps_band_stats.npz", mean=m, std=s)

