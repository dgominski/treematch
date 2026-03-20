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


class SPOTCountingDataset(torch.utils.data.Dataset):
    def __init__(self, imsize, split, preload=False, **kwargs):
        assert split in ["train", "test"], "Invalid split"
        self.split = split

        band_stats = np.load("data/spot_band_stats.npz")
        self.crop = A.Compose([
            A.PadIfNeeded(min_height=imsize, min_width=imsize, border_mode=0, fill=0),
            A.CenterCrop(height=imsize, width=imsize),
        ],
            additional_targets={'valid': 'mask'},
            seed=42
        )
        self.transform = T.Compose([
            T.Normalize(mean=band_stats['mean'].tolist(), std=band_stats['std'].tolist())
        ])

        pt_dir = f"/data/Open-Canopy/datasets/count/pt/{self.split}/"
        self.pts = glob.glob(os.path.join(pt_dir, "*.pt"))

        self.preloaded = False
        if preload:
            self.preload()

        self.nbands = 4

    def __getitem__(self, index):
        if self.preloaded:
            data = self.data[index]
        else:
            data = torch.load(self.pts[index])

        target = data['labels']
        inp = data['spot']
        # apply random crop
        augmented = self.crop(image=inp.permute(1, 2, 0).numpy(),
                              mask=target.permute(1, 2, 0).numpy().astype(np.uint8),
                              valid=data['valid'].permute(1, 2, 0).numpy().astype(np.uint8))
        raw_image = np.transpose(augmented['image'], (2, 0, 1))
        image = self.transform(torch.tensor(raw_image, dtype=torch.float32))
        valid = torch.from_numpy(augmented['valid'].transpose(2, 0, 1)).float()
        target = torch.from_numpy(augmented['mask'].transpose(2, 0, 1)).float()

        inp = torch.cat([image, valid], dim=0)  # append valid mask as last channel
        # count over cropped target
        # gt_count = (target * inp[-1:, :, :]).sum()  # multiply by valid mask
        return inp, target

    def __len__(self):
        return len(self.pts)

    def preload(self):
        self.data = []
        for idx in range(len(self.pts)):
            data = torch.load(self.pts[idx])
            self.data.append(data)
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


class SPOTUnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, imsize, preload=False, **kwargs):
        self.geometries = gpd.read_file("/data/Open-Canopy/datasets/count/geometries.geojson")

        self.point_fps = glob.glob("/data/Open-Canopy/datasets/count/pseudolabels/*.gpkg")
        # remove unexisting ids
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
        spot_fp = f"/data/Open-Canopy/datasets/canopy_height/{year}/spot/compressed_pansharpened_{tile_id}.tif"
        point_fp = f"/data/Open-Canopy/datasets/count/pseudolabels/cropped_{lidar_crop_id}.gpkg"

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
        # count over cropped target
        # gt_count = (target * inp[-1:, :, :]).sum()  # multiply by valid mask
        return inp, target

    def __len__(self):
        return len(self.geometries)

    def loader(self, batch_size, num_workers):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


def prepare():
    import numpy as np
    import geopandas as gpd
    import shapely as shp
    from shapely.geometry import box
    import rasterio.mask

    spot_dir = "/data/Open-Canopy/datasets/canopy_height/2021/spot/"
    spot_fps = glob.glob(os.path.join(spot_dir, "*.tif"))
    spot_fps += glob.glob(os.path.join(spot_dir.replace("2021", "2022"), "*.tif"))
    spot_fps += glob.glob(os.path.join(spot_dir.replace("2021", "2023"), "*.tif"))
    bounds = []
    for fp in spot_fps:
        with rasterio.open(fp) as src:
            bounds.append(src.bounds)
    bounds = np.array(bounds)
    geometries = [box(b[0], b[1], b[2], b[3]) for b in bounds]
    bounds_gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:2154")

    plot_centers = gpd.read_file("/scratch/france/nfi/all_plots.gpkg")

    # join, how many plots are within spot images
    joined = gpd.sjoin(plot_centers.to_crs("EPSG:2154"), bounds_gdf, how='inner', predicate='within')
    print(f"Number of plots within SPOT images: {len(joined)} / {len(plot_centers)}")

    points = gpd.read_file("/scratch/france/nfi/indiv_points/all_points.gpkg")
    # iterate over spot images and save 50m x 50m rectangles around each plot center
    out_dir = "/data/Open-Canopy/datasets/count/pt/"
    for i, row in tqdm(joined.iterrows(), total=len(joined)):
        plot_id = row['plot_id']
        center = row['geometry']
        minx, miny = center.x - 48, center.y - 48  # in meters
        maxx, maxy = center.x + 48, center.y + 48  # in meters
        plot_box = box(minx, miny, maxx, maxy)
        out_fp = os.path.join(out_dir, f"plot_{plot_id}.tif")
        if os.path.exists(out_fp):
            continue
        spot_idx = row['index_right']
        spot_fp = spot_fps[spot_idx]
        with rasterio.open(spot_fp) as src:
            out_image, out_transform = rasterio.mask.mask(src, [plot_box], crop=True)
            out_meta = src.meta.copy()

        # # save canopy height and point map
        # out_image = np.vstack([out_image, pmap[np.newaxis, ...], rasterized_disk[np.newaxis, ...]])
        #
        # out_meta.update({
        #     "driver": "GTiff",
        #     "count": out_image.shape[0],
        #     "height": out_image.shape[1],
        #     "width": out_image.shape[2],
        #     "transform": out_transform
        # })
        # with rasterio.open(out_fp, "w", **out_meta) as dest:
        #     dest.write(out_image)


def count_sweep():
    # find optimal height threshold and min distance for peak detection
    dataset = SPOTCountingDataset(imsize=64)


def print_stats():
    points = gpd.read_file("/data/Open-Canopy/datasets/count/points.gpkg")
    train_plots = gpd.read_file("/data/Open-Canopy/datasets/count/train_plots.gpkg")
    test_plots = gpd.read_file("/data/Open-Canopy/datasets/count/test_plots.gpkg")

    # # 80/20 train/test split at plot level
    # plots = gpd.read_file("/data/Open-Canopy/datasets/count/plots.gpkg")
    # train_plots = plots.sample(frac=0.5, random_state=42)
    # test_plots = plots[~plots['plot_id'].isin(train_plots['plot_id'])]
    train_points = points[points['plot_id'].isin(train_plots['plot_id'])]
    test_points = points[points['plot_id'].isin(test_plots['plot_id'])]
    # # save to gpkg
    # train_plots.to_file("/data/Open-Canopy/datasets/count/train_plots.gpkg", driver="GPKG")
    # test_plots.to_file("/data/Open-Canopy/datasets/count/test_plots.gpkg", driver="GPKG")

    # print stats
    print(f"Train plots: {len(train_plots)}, Train points: {len(train_points)}")
    print(f"Test plots: {len(test_plots)}, Test points: {len(test_points)}")
    train_area = len(train_plots) * (15 * 15 * np.pi) / 1e6  # km²
    test_area = len(test_plots) * (15 * 15 * np.pi) / 1e6  # km²
    print(f"Train area (km²): {train_area}, Test area (km²): {test_area}")

    geometries = gpd.read_file("/data/Open-Canopy/datasets/canopy_height/geometries.geojson")
    # filter to 2021
    geometries = geometries[geometries['lidar_year'] == 2021]
    total_area = geometries.geometry.area.sum() / 1e6  # km²
    print(f"Total SPOT 2021 image area (km²): {total_area}")
    exit()


def crop_to_geometries(target_dir, out_dir):
    import rasterio.mask
    from shapely.geometry import box
    geometries = gpd.read_file("/data/Open-Canopy/datasets/canopy_height/geometries.geojson")
    rasters = glob.glob(os.path.join(target_dir, "*.tif"))

    #get raster bounds
    bounds_gdf = []
    for fp in rasters:
        with rasterio.open(fp) as src:
            bounds = src.bounds
        geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        bounds_gdf.append({'geometry': geom, 'fp': fp})
    bounds_gdf = gpd.GeoDataFrame(bounds_gdf, crs="EPSG:2154")

    for i, geom in tqdm(geometries.iterrows(), total=len(geometries)):
        raster_fp = bounds_gdf.sjoin(gpd.GeoDataFrame(geometry=[geom.geometry], crs="EPSG:2154"), how='inner', predicate='intersects')
        if len(raster_fp) == 0:
            continue
        if len(raster_fp) > 1:
            # take largest intersection
            raster_fp['intersection_area'] = raster_fp.geometry.intersection(geom.geometry).area
            raster_fp = raster_fp.sort_values(by='intersection_area', ascending=False)
            fp = raster_fp.iloc[0]['fp']
        else:
            fp = raster_fp.iloc[0]['fp']
        out_fp = os.path.join(out_dir, f"cropped_{i}.tif")
        if os.path.exists(out_fp):
            continue
        with rasterio.open(fp) as src:
            try:
                out_image, out_transform = rasterio.mask.mask(src, [geom.geometry, ], crop=True)
            except Exception as e:
                print(f"Error cropping {fp} with geometry {i}: {e}")
                continue
            out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        with rasterio.open(out_fp, "w", **out_meta) as dest:
            dest.write(out_image)



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import matplotlib.pyplot as plt
    from itertools import cycle
    # prepare()

    GSD = 1.5  # meters
    band_stats = np.load("data/spot_band_stats.npz")
    mean = band_stats['mean'].tolist()
    std = band_stats['std'].tolist()

    train_dataset = SPOTCountingDataset(imsize=64, split="train", preload=True)
    test_dataset = SPOTCountingDataset(imsize=64, split="test", preload=True)

    dataset_pseudolabels = SPOTUnlabeledDataset(imsize=64)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    pseudolabels_loader = dataset_pseudolabels.loader(batch_size=16, num_workers=4)
    for batch_strong, batch_weak in zip(train_loader, cycle(pseudolabels_loader)):
        inp_strong, target_strong = batch_strong
        inp_weak, target_weak = batch_weak

        unnormed = inp_strong[:, :4] * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
        unnormed_weak = inp_weak[:, :4] * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)

        print("iter")
        fig, axs = plt.subplots(1, 4, figsize=(12, 4))
        axs[0].imshow(target_strong[0, 0, :, :].numpy(), cmap='viridis')
        axs[0].set_title("Strongly labeled count map")
        axs[1].imshow(unnormed[0, :3].permute(1, 2, 0).numpy().astype(np.uint8))
        axs[1].set_title("Strongly labeled image")
        axs[2].imshow(target_weak[0, 0, :, :].numpy(), cmap='viridis')
        axs[2].set_title("Pseudolabeled count map")
        axs[3].imshow(unnormed_weak[0, :3].permute(1, 2, 0).numpy().astype(np.uint8))
        axs[3].set_title("Pseudolabeled image")
        plt.show()



    # total_area = 0
    # nb_trees = 0
    # dataset = SPOTCountingDataset(imsize=64)
    # for i in range(len(dataset)):
    #     a = dataset[i]
    #     valid = a[0][-1, :, :]
    #     area = (valid.sum().item()) * (GSD ** 2) / 1e4
    #     total_area += area
    #     target = a[1][0, :, :]
    #     nb_trees += target.sum().item()
    # print(f"Total sampled area (ha): {total_area}")
    # print(f"Total number of trees: {nb_trees}")
    #
    # def count_worker(fp):
    #     gdf = gpd.read_file(fp)
    #     return len(gdf)
    #
    # geometries = gpd.read_file("/data/Open-Canopy/datasets/count/geometries.geojson")
    # geometries.to_file("/scratch/tmp/geometries.geojson", driver="GeoJSON")
    # fps = [os.path.join("/data/Open-Canopy/datasets/count/pseudolabels/", f"cropped_{gid}.gpkg") for gid in geometries['crop_id'].astype(str).values]
    # with Pool(32) as pool:
    #     counts = list(tqdm(pool.imap(count_worker, fps), total=len(fps), desc="Counting pseudolabels"))
    # nb_trees = sum(counts)
    # print(f"Total unlabeled number of trees: {nb_trees}")
    # total_area = geometries.geometry.area.sum() / 1e4  # ha
    # print(f"Total unlabeled sampled area (ha): {total_area}")

    # dataset = SPOTUnlabeledDataset(imsize=600)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    # for batch in tqdm(loader, total=len(loader), desc="Unlabeled data stats"):
    #     target = batch[1][:, 0, :, :]
    #     valid = batch[0][:, -1, :, :]
    #     area = (valid.sum().item()) * (GSD ** 2) / 1e4




