##############################
#Part 2: LCC Batch Processing#
#Maintainer: Christopher Chan#
#Version: 0.0.2              #
#Date: 2025-02-14            #
##############################

import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import List
from PIL import Image
from osgeo import gdal
from rasterio.mask import mask
from rasterio.enums import Resampling
from glob import glob


# path setup
BASE_PATH = Path(os.getcwd())
data_raw = (BASE_PATH/'../../data/01_raw').resolve()
data_intermediate = (BASE_PATH/'../../data/02_intermediate').resolve()
data_model_output = (BASE_PATH/'../../data/07_model_output').resolve()
docs_path = (BASE_PATH/'../../docs').resolve()

# Setup logging
log_dir = Path(f'{docs_path}/Part_2/logs')
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'batch_process_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will print to console too
    ]
)

# load data
def load_raster(raster_path: Path) -> List[str]:
    raster_list = []
    for path in glob(f'{raster_path}/*.tif'):
            ds = gdal.Open(str(path))
            raster_list.append(ds.GetDescription())

    return raster_list

def load_vector(vector_path: Path) -> List[tuple[gpd.GeoDataFrame, str]]:
    vector_list = []
    for path in glob(f'{vector_path}/*.geojson'):
        ds = gpd.read_file(path)
        filename = Path(path).stem  # Gets filename without extension
        vector_list.append((ds, filename))
    return vector_list

def batch_clip(raster_list: List[str], vector_list: List[tuple[gpd.GeoDataFrame, str]]) -> List[rio.DatasetReader]:
    clip_rasterls = []
    
    # Create output directory if it doesn't exist
    out_dir = Path(f'{data_intermediate}/raster/Part_2')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(raster_list)} rasters with {len(vector_list)} vector files")
    
    for raster in raster_list:
        print(f"Processing raster: {raster}")
        year = raster.split('_')[2].split('.')[0]

        for gdf, vector_path in vector_list:
            print(f"Clipping with vector: {vector_path}")

            # Reproject
            if gdf.crs != 'EPSG:32630':
                 gdf_32630 = gdf.to_crs(epsg=32630)
            else:
                 gdf_32630 = gdf
                 
            id = vector_path

            with rio.open(raster) as src:
                try:
                    clipped_image, clipped_transform = mask(src, gdf_32630.geometry, crop=True)
                    clipped_image = np.where((clipped_image == 0) | (clipped_image == 10), 0, clipped_image).astype(np.int8)

                    profile = src.profile.copy()
                    profile.update({
                        "height": clipped_image.shape[1],
                        "width": clipped_image.shape[2],
                        "transform": clipped_transform,
                        "driver": "GTiff",
                        "dtype": "int8"
                    })

                    with rio.open(f'{data_intermediate}/raster/Part_2/{id}_{year}.tif', 'w', **profile) as dst:
                        print(f"Saving to: {data_intermediate}/raster/Part_2/{id}_{year}.tif")
                        dst.write(clipped_image)

                    clip_rasterls.append(clipped_image)

                except ValueError as e:
                    logging.error(f"Error clipping raster {raster} with vector {vector_path}: {e}")
                    continue

    logging.info(f"Completed processing. Created {len(clip_rasterls)} clipped rasters")
    return clip_rasterls

def main():
    raster_list = load_raster(Path(f'{data_raw}/raster'))
    vector_list = load_vector(Path(f'{data_raw}/vector/Part_2'))
    clip_rasterls = batch_clip(raster_list, vector_list)

if __name__ == "__main__":
    main()