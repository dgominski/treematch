import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
import os
import rasterio
from torch.multiprocessing import Process, Queue, Event
import torch.multiprocessing as mp
from itertools import product
from rasterio.windows import Window, from_bounds
import warnings
from utils import split_tensor
from models.backbones import UNetCounter
import hydra
import geopandas as gpd
from torchvision.transforms import Normalize


# fix to avoid subprocesses getting stuck due to torch internal CPU multithreading
torch.set_num_threads(1)
mp.set_start_method("spawn", force=True)


def get_tiles(ps_fps, s2_fps, s1_fps, limit):
    tiles = []
    for ps_fp, s2_fp, s1_fp in zip(ps_fps, s2_fps, s1_fps):
        ps = rasterio.open(ps_fp)
        width = ps.width
        height = ps.height
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

                tiles.append([ps_fp, s2_fp, s1_fp, window, nb])

        else:
            tiles.append([ps_fp, s2_fp, s1_fp, from_bounds(*ps.bounds, ps.transform), None])

    return tiles


def reader(cfg, reading_queue, loaded_queue, reading_log, end_of_reading_queue, end_of_writing_queue):
    while True:
        data = reading_queue.get()
        if data is None:
            if not end_of_reading_queue.is_set() and reading_queue.empty():
                # process finishing queue
                loaded_queue.put(None)
                end_of_reading_queue.set()
            # waiting for all processes to finish
            end_of_writing_queue.wait()
            break
        ps_fp, s2_fp, s1_fp, window, window_nb = data
        out_name = os.path.split(ps_fp)[1]
        # make window nb a two digit string
        window_nb = f"{window_nb:02d}" if window_nb is not None else ""
        out_fp = os.path.join(cfg.output_dir, os.path.splitext(out_name)[0] + f"_{window_nb}" + ".tif") # integrate window parameters in filename for merging later
        if os.path.exists(out_fp):
            print(f"skipping {out_fp}, already exists")
            continue
        try:
            with rasterio.open(ps_fp) as src:
                ps_raster = rasterio.open(ps_fp)
                ps_array = ps_raster.read(window=window)
        except Exception as e:
            warnings.warn(f"Could not read raster {ps_fp}, skipping. Error: {e}")
            continue
        ps_array = ps_array.astype(float)

        # read s2
        try:
            s2_raster = rasterio.open(s2_fp)
            s2_window = rasterio.windows.from_bounds(*rasterio.windows.bounds(window, ps_raster.transform), transform=s2_raster.transform)
            s2_array = s2_raster.read(window=s2_window)
        except Exception as e:
            warnings.warn(f"Could not read raster {s2_fp}, skipping. Error: {e}")
            continue
        s2_array = s2_array.astype(float)

        # read s1
        try:
            s1_raster = rasterio.open(s1_fp)
            s1_window = rasterio.windows.from_bounds(*rasterio.windows.bounds(window, ps_raster.transform), transform=s1_raster.transform)
            s1_array = s1_raster.read(window=s1_window)
            s1_array = s1_array.astype(float)
        except Exception as e:
            warnings.warn(f"Could not read raster {s1_fp}, skipping. Error: {e}")
            continue
        s1_array = s1_array.astype(float)

        with torch.no_grad():
            interp_device = torch.device(cfg.interp_device)
            s2_array = torch.nn.functional.interpolate(torch.from_numpy(s2_array).unsqueeze(0).to(interp_device),
                                                       size=(ps_array.shape[1], ps_array.shape[2]), mode='bilinear',
                                                       align_corners=False)[0]
            s1_array = torch.nn.functional.interpolate(torch.from_numpy(s1_array).unsqueeze(0).to(interp_device),
                                                       size=(ps_array.shape[1], ps_array.shape[2]), mode='bilinear',
                                                       align_corners=False)[0]

            data = torch.cat([torch.from_numpy(ps_array).to(interp_device), s2_array, s1_array], dim=0)[None, ...].cpu()

        reading_log.set_description_str(f"READER --- input array {ps_fp} shape {list(data.shape)} read from disk")

        # make tensor dimensions multiples of 2
        if data.shape[2] % 2 != 0:
            data = torch.nn.functional.pad(data, (0, 0, 0, 1), mode='constant', value=0)
        if data.shape[3] % 2 != 0:
            data = torch.nn.functional.pad(data, (0, 1, 0, 0), mode='constant', value=0)

        try:
            patches, t_size, p0, p1, offsets = split_tensor(data, patch_size=cfg.imsize, overlap=0)
        except Exception as e:
            warnings.warn(f"Could not split raster {out_fp}, skipping. Error: {e}")
            del data
            continue

        reading_log.set_description_str(f"READER --- input array {ps_fp} split into {patches.shape[0]} patches, putting into processing queue")

        if patches.shape[0] != offsets.shape[0]:
            print(f"### Raster {out_fp} could not be split properly into patches. Skipping.")
        del data
        loaded_queue.put([patches.cpu(), t_size, p0, p1, offsets, window, ps_raster.profile, out_fp])


def processor(cfg, loaded_queue, writing_queue, model, normalizer, prediction_log, end_of_writing_queue):
    while True:
        data = loaded_queue.get()
        if data is None:
            [writing_queue.put(None) for _ in range(cfg.num_writers)]
            break
        patches, t_size, p0, p1, offsets, window, profile, out_fp = data
        del data

        device = torch.device(cfg.pred_device)

        # merge and create dataset
        dataset = torch.utils.data.TensorDataset(patches, offsets)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=cfg.num_workers)

        forward_log = tqdm.tqdm(total=len(loader), position=3, leave=False, desc=f"DCNN   --- Forward pass for raster {os.path.split(out_fp)[-1]}")
        with torch.no_grad():
            # Run predictions for tiles and accumulate them
            counts = []
            for batch in loader:
                patch, offset = batch
                transformed = normalizer(patch).float().to(device)
                patch_counts = model(transformed)
                counts.append(patch_counts.cpu())
                forward_log.update()

        counts = torch.cat(counts, dim=0)

        prediction_log.update()

        writing_queue.put([counts, t_size, p0, p1, window, profile, out_fp])

    end_of_writing_queue.wait()


def writer(cfg, writing_queue, end_of_writing_queue, saving_log, writing_file):
    while True:
        data = writing_queue.get()
        if data is None:
            if not end_of_writing_queue.is_set() and writing_queue.empty():
                # process finishing queue
                end_of_writing_queue.set()
            end_of_writing_queue.wait()
            break

        counts, t_size, p0, p1, window, profile, out_fp = data
        del data

        # reconstruct large tensor of probabilities
        try:
            width = np.ceil(t_size[1] / cfg.imsize).astype(int)
            height = np.ceil(t_size[0] / cfg.imsize).astype(int)
            counts = counts.reshape(height, width)
        except Exception as e:
            warnings.warn(f"Could not rebuild output raster for {out_fp}, skipping. Error: {e}")
            del counts, t_size, p0, p1, window, profile, out_fp
            continue

        try:
            # save
            profile = {
                "driver": "GTiff",
                "dtype": "float32",
                "count": 1,
                "height": counts.shape[0],
                "width": counts.shape[1],
                "transform": rasterio.transform.from_bounds(*rasterio.windows.bounds(window, profile["transform"]),
                                                            counts.shape[1], counts.shape[0]),
                "crs": profile["crs"],
            }
            with rasterio.open(out_fp, "w", **profile) as dst:
                dst.write(counts.numpy(), 1)
        except Exception as e:
            warnings.warn(f"Could not save output raster for {out_fp}, skipping. Error: {e}")
            del counts, t_size, p0, p1, window, profile, out_fp
            continue

        saving_log.set_description_str(f"WRITER --- saved prediction raster {out_fp}")
        writing_file.set()

        del t_size, p0, p1, window, profile, out_fp, counts


def queue_monitor(prediction_log, loaded_queue_log, writing_queue_log, end_of_writing_queue, loaded_queue, writing_queue, writing_file):
    while not end_of_writing_queue.is_set():
        loaded_queue_log.n = loaded_queue.qsize()
        writing_queue_log.n = writing_queue.qsize()
        loaded_queue_log.refresh()
        writing_queue_log.refresh()
        if writing_file.is_set():
            writing_file.clear()
            prediction_log.update()


def get_prediction_fps(cfg):
    ps_gdf = gpd.read_file(cfg.ps_gpkg)
    s2_gdf = gpd.read_file(cfg.s2_gpkg)
    s1_gdf = gpd.read_file(cfg.s1_gpkg)
    prediction_fps = []
    for idx, row in ps_gdf.iterrows():
        # find s2 and s1 with max overlap
        ps_geom = row.geometry
        s2_overlaps = s2_gdf.intersection(ps_geom).area
        s1_overlaps = s1_gdf.intersection(ps_geom).area
        if s2_overlaps.max() == 0 or s1_overlaps.max() == 0:
            continue
        s2_idx = s2_overlaps.idxmax()
        s1_idx = s1_overlaps.idxmax()
        ps_fp = os.path.join(cfg.ps_dir, row['fp_name']+".jp2")
        s2_fp = os.path.join(cfg.s2_dir, s2_gdf.loc[s2_idx, 'fp_name']+".jp2")
        s1_fp = os.path.join(cfg.s1_dir, s1_gdf.loc[s1_idx, 'fp_name']+".tif")
        if all(os.path.exists(fp) for fp in [ps_fp, s2_fp, s1_fp]):
            prediction_fps.append([ps_fp, s2_fp, s1_fp])
    return prediction_fps


@hydra.main(version_base=None, config_path="conf", config_name="predict")
def predict(cfg):
    TILE_MAX_SIZE = 10000
    loaded_queue_maxsize = 1
    num_writers = 2
    writing_queue_maxsize = 1

    device = torch.device(cfg.pred_device)

    model = UNetCounter(cfg, in_channels=20, pretrained=False)
    ckpt = torch.load(cfg.ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    band_stats = np.load("data/ps_band_stats.npz")
    normalizer = Normalize(mean=band_stats['mean'].tolist(), std=band_stats['std'].tolist())

    in_fps = get_prediction_fps(cfg)[::-1]
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    target_images = np.array(in_fps).tolist()

    ps_fps = [ti[0] for ti in target_images]
    s2_fps = [ti[1] for ti in target_images]
    s1_fps = [ti[2] for ti in target_images]
    tiles = get_tiles(ps_fps, s2_fps, s1_fps, limit=TILE_MAX_SIZE)

    reading_queue = Queue()
    [reading_queue.put(ti) for ti in tiles]
    reading_queue.put(None)

    end_of_reading_queue = Event()
    end_of_writing_queue = Event()
    writing_file = Event()

    prediction_log = tqdm.tqdm(total=len(tiles), position=1, desc="Running predictions for all target rasters", disable=True)
    reading_log = tqdm.tqdm(total=len(tiles), position=2, bar_format='{desc}', disable=True)
    saving_log = tqdm.tqdm(total=len(tiles), position=4, bar_format='{desc}', disable=True)
    loaded_queue_log = tqdm.tqdm(total=loaded_queue_maxsize, position=5, desc="Loading queue monitoring",
                                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ', disable=True)
    writing_queue_log = tqdm.tqdm(total=writing_queue_maxsize, position=6, desc="Writing queue monitoring",
                                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ', disable=True)

    loaded_queue = Queue(maxsize=loaded_queue_maxsize)  # limiting the loading queue size
    reader_p = Process(target=reader, args=(cfg, reading_queue, loaded_queue, reading_log, end_of_reading_queue, end_of_writing_queue))
    reader_p.start()

    writing_queue = Queue(maxsize=writing_queue_maxsize)  # limiting the writing queue size
    writers = []
    for i in range(num_writers):
        writer_p = Process(target=writer, args=(cfg, writing_queue, end_of_writing_queue, saving_log, writing_file))
        writer_p.daemon = True
        writer_p.start()
        writers.append(writer_p)

    queue_monitor_p = Process(target=queue_monitor, args=(prediction_log, loaded_queue_log, writing_queue_log, end_of_writing_queue, loaded_queue, writing_queue,writing_file))
    queue_monitor_p.daemon = True
    queue_monitor_p.start()

    processor(cfg, loaded_queue, writing_queue, model, normalizer, prediction_log, end_of_writing_queue)

    queue_monitor_p.join()
    reader_p.join()
    [writer_p.join() for writer_p in writers]


if __name__ == '__main__':
    predict()