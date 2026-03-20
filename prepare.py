import glob
import shutil

import geopandas as gpd
import pandas as pd
import shapely as shp
import tqdm
import pyproj
import os
import multiprocessing
from pyproj import Transformer
from PIL import Image
import subprocess
import rasterio
import numpy as np
from shapely.geometry import box
import seaborn as sns
import matplotlib.pyplot as plt
# from data.plot_dataset import extract_species
from rasterio import merge

S2_DIR = "/bigdata/users/Maurice/Study_3/Data/Training_data/Images/S2_10m_2019"
S1_DIR = "/bigdata/users/Maurice/Study_3/Data/Training_data/Images/S1_10m_2019"
PS_DIR = "/bigdata/users/Maurice/Study_3/Data/Training_data/Images/Planet_3m_2019"


def read_raster_bounds(dir=None, target_crs="EPSG:4326"):
    all_raster_paths = []
    for r, d, f in os.walk(dir):
        for file in f:
            if ".tif" in file or "jp2" in file:
                all_raster_paths.append(os.path.join(r, file))
    rgb_raster_bounds = []
    raster_paths = []
    for r in tqdm.tqdm(all_raster_paths, total=len(all_raster_paths), desc=f"Reading raster bounds in {dir}"):
        try:
            bounds = rasterio.open(r).bounds
        except:
            continue
        crs = rasterio.open(r).crs
        # transform to target crs
        if crs is None:
            print(r)
        transformer = pyproj.Transformer.from_crs(crs, target_crs, always_xy=True)
        bounds = transformer.transform(bounds[0], bounds[1]) + transformer.transform(bounds[2], bounds[3])
        rgb_raster_bounds.append(bounds)
        raster_paths.append(r)
    rgb_raster_bounds_gdf = gpd.GeoDataFrame(geometry=[shp.geometry.box(*b) for b in rgb_raster_bounds], crs=target_crs)
    rgb_raster_bounds_gdf['path'] = raster_paths
    return rgb_raster_bounds_gdf


def extract_data():
    from rasterio.features import rasterize
    rectangle_gdf = gpd.read_file("data/rectangles.gpkg")
    points_gdf = gpd.read_file("data/labels.gpkg")
    s2_raster_bounds = gpd.read_file("data/s2_bounds.gpkg")
    s1_raster_bounds = gpd.read_file("data/s1_bounds.gpkg")
    ps_raster_bounds = gpd.read_file("data/ps_bounds.gpkg")

    # iterate over rectangles
    for i, row in tqdm.tqdm(rectangle_gdf.iterrows(), total=rectangle_gdf.shape[0]):
        rect_geom = row.geometry
        rect_folder = f"data/tiles"
        os.makedirs(rect_folder, exist_ok=True)

        # s2_output_path = os.path.join(rect_folder, f"{i}_s2.tif")
        # s1_output_path = os.path.join(rect_folder, f"{i}_s1.tif")
        # ps_output_path = os.path.join(rect_folder, f"{i}_ps.tif")
        # labels_output_path = os.path.join(rect_folder, f"{i}_labels.tif")
        # if all([os.path.exists(p) for p in [s2_output_path, s1_output_path, ps_output_path, labels_output_path]]):
        #     continue
        #
        # # extract S2 rasters
        # s2_overlaps = s2_raster_bounds[s2_raster_bounds.intersects(rect_geom)]
        # s2_raster_paths = s2_overlaps['path'].tolist()
        # s2_rasters = [rasterio.open(p) for p in s2_raster_paths]
        # if len(s2_rasters) > 0:
        #     try:
        #         s2_merged, s2_transform = merge.merge(s2_rasters, bounds=rect_geom.bounds)
        #         s2_meta = s2_rasters[0].meta.copy()
        #         s2_meta.update({
        #             "height": s2_merged.shape[1],
        #             "width": s2_merged.shape[2],
        #             "transform": s2_transform
        #         })
        #         s2_output_path = os.path.join(rect_folder, f"{i}_s2.tif")
        #         with rasterio.open(s2_output_path, "w", **s2_meta) as dest:
        #             dest.write(s2_merged)
        #         for r in s2_rasters:
        #             r.close()
        #     except Exception as e:
        #         print(f"Error merging S2 rasters for rectangle {i}: {e}")
        #
        # # extract S1 rasters
        # s1_overlaps = s1_raster_bounds[s1_raster_bounds.intersects(rect_geom)]
        # s1_raster_paths = s1_overlaps['path'].tolist()
        # s1_rasters = [rasterio.open(p) for p in s1_raster_paths]
        # if len(s1_rasters) > 0:
        #     try:
        #         s1_merged, s1_transform = merge.merge(s1_rasters, bounds=rect_geom.bounds)
        #         s1_meta = s1_rasters[0].meta.copy()
        #         s1_meta.update({
        #             "height": s1_merged.shape[1],
        #             "width": s1_merged.shape[2],
        #             "transform": s1_transform
        #         })
        #         s1_output_path = os.path.join(rect_folder, f"{i}_s1.tif")
        #         with rasterio.open(s1_output_path, "w", **s1_meta) as dest:
        #             dest.write(s1_merged)
        #         for r in s1_rasters:
        #             r.close()
        #     except Exception as e:
        #         print(f"Error merging S1 rasters for rectangle {i}: {e}")
        #
        # extract PS rasters
        ps_overlaps = ps_raster_bounds[ps_raster_bounds.intersects(rect_geom)]
        ps_raster_paths = ps_overlaps['path'].tolist()
        ps_rasters = [rasterio.open(p) for p in ps_raster_paths]
        if len(ps_rasters) > 0:
            try:
                ps_merged, ps_transform = merge.merge(ps_rasters, bounds=rect_geom.bounds)
                ps_meta = ps_rasters[0].meta.copy()
                ps_meta.update({
                    "height": ps_merged.shape[1],
                    "width": ps_merged.shape[2],
                    "transform": ps_transform
                })
                ps_output_path = os.path.join(rect_folder, f"{i}_ps.tif")
            #     with rasterio.open(ps_output_path, "w", **ps_meta) as dest:
            #         dest.write(ps_merged)
            #     for r in ps_rasters:
            #         r.close()
            except Exception as e:
                print(f"Error merging PS rasters for rectangle {i}: {e}")
        #
        # # rasterize labels, if multiple on same pixel sum them
        # points_in_rect = points_gdf[points_gdf.intersects(rect_geom)]
        # shapes = ((geom, 1) for geom in points_in_rect.geometry)
        # out_shape = ps_merged.shape[1:]
        # rasterized_labels = rasterize(shapes, out_shape=out_shape, transform=ps_transform, fill=0, all_touched=True, dtype='uint16')
        # labels_output_path = os.path.join(rect_folder, f"{i}_labels.tif")
        # try:
        #     with rasterio.open(labels_output_path, "w", driver='GTiff', height=rasterized_labels.shape[0], width=rasterized_labels.shape[1], count=1, dtype=rasterized_labels.dtype, crs="EPSG:4326", transform=rasterio.transform.from_bounds(*rect_geom.bounds, rasterized_labels.shape[1], rasterized_labels.shape[0])) as dest:
        #         dest.write(rasterized_labels, 1)
        # except Exception as e:
        #     print(f"Error writing rasterized labels for rectangle {i}: {e}")

        # rasterize annotation rectangle
        shapes = ((rect_geom, 1),)
        out_shape = ps_merged.shape[1:]
        rasterized_rect = rasterize(shapes, out_shape=out_shape, transform=ps_transform, fill=0, all_touched=True, dtype='uint8')
        rect_output_path = os.path.join(rect_folder, f"{i}_rect.tif")
        try:
            with rasterio.open(rect_output_path, "w", driver='GTiff', height=rasterized_rect.shape[0], width=rasterized_rect.shape[1], count=1, dtype=rasterized_rect.dtype, crs="EPSG:4326", transform=rasterio.transform.from_bounds(*rect_geom.bounds, rasterized_rect.shape[1], rasterized_rect.shape[0])) as dest:
                dest.write(rasterized_rect, 1)
        except Exception as e:
            print(f"Error writing rasterized rectangle for rectangle {i}: {e}")

if __name__ == "__main__":
    extract_data()
