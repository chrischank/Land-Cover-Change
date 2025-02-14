##############################
#Part 2: LCC Batch Processing#
#Maintainer: Christopher Chan#
#Version: 0.0.3              #
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

def batch_clip(raster_list: List[str], vector_list: List[tuple[gpd.GeoDataFrame, str]]) -> dict:
    clip_rasterDICT = {}

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
                    # Cloud mask
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

                    # Initialize dictionary entry for this ID if it doesn't exist
                    if id not in clip_rasterDICT:
                        clip_rasterDICT[id] = {}
                    
                    # Add the year and raster to the ID's dictionary
                    clip_rasterDICT[id][year] = {"raster": clipped_image}

                except ValueError as e:
                    logging.error(f"Error clipping raster {raster} with vector {vector_path}: {e}")
                    continue

    logging.info(f"Completed processing. Created {len(clip_rasterDICT)} clipped rasters")
    return clip_rasterDICT

def LC_raster_plot(raster_dict: dict) -> None:
    # Create output directory if it doesn't exist
    out_dir = Path(f'{docs_path}/Part_2/plots')
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 13):
        id_key = f"id_{i}"
        if id_key not in raster_dict:
            continue
            
        images = {"2020": None, "2021": None, "2022": None}
        
        # Collect all images for this ID
        for year in raster_dict[id_key]:
            images[year] = raster_dict[id_key][year]["raster"][0]  # Get first band
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Land Cover for {id_key}")
        
        # Plot each year
        for idx, (year, img) in enumerate(images.items()):
            if img is not None:
                im = axes[idx].imshow(img)
                axes[idx].set_title(f"Year {year}")
                
                # Add colorbar
                arr = np.arange(0, 12)
                fig.colorbar(im, ax=axes[idx], orientation='horizontal', 
                           fraction=0.09, pad=0.09, ticks=arr,
                           boundaries=arr)
        
        plt.savefig(f"{docs_path}/Part_2/plots/{id_key}_timeseries.png")
        plt.close()  # Close the figure to free memory

def change_detection(raster_dict: dict) -> None:
    out_dir = Path(f'{docs_path}/Part_2/CD_plots')
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 13):
        id_key = f"id_{i}"
        if id_key not in raster_dict:
            continue
        images = {"2020": None, "2021": None, "2022": None}
        
        # Collect all images for this ID
        for year in raster_dict[id_key]:
            images[year] = raster_dict[id_key][year]["raster"][0]

        try:
            # Get maximum dimensions
            max_height = max(img.shape[0] for img in images.values() if img is not None)
            max_width = max(img.shape[1] for img in images.values() if img is not None)

            # Pad images to match largest dimensions
            padded_images = {}
            for year, img in images.items():
                if img is not None:
                    pad_height = max_height - img.shape[0]
                    pad_width = max_width - img.shape[1]
                    padded_images[year] = np.pad(
                        img,
                        pad_width=((0, pad_height), (0, pad_width)),
                        mode="constant",
                        constant_values=0
                    )

            # Create change detection arrays
            LCC2020_2021 = np.char.add(padded_images["2020"].astype(str), padded_images["2021"].astype(str)).astype(np.float16)
            LCC2021_2022 = np.char.add(padded_images["2021"].astype(str), padded_images["2022"].astype(str)).astype(np.float16)

            def _LCC_str_ndarray(LC1: np.ndarray, LC2: np.ndarray) -> np.ndarray:
                temp_array = np.char.add(LC1.astype(str), ":")
                return np.char.add(temp_array, LC2.astype(str))
            
            LCC2020_2021_str = _LCC_str_ndarray(padded_images["2020"], padded_images["2021"])
            LCC2021_2022_str = _LCC_str_ndarray(padded_images["2021"], padded_images["2022"])

            # Plot
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f"Land Cover Change for {id_key}")
            
            LCCPlot_2020_2021 = ax[0].imshow(LCC2020_2021)
            LCCPlot_2021_2022 = ax[1].imshow(LCC2021_2022)

            ax[0].set_title("2020-2021")
            ax[1].set_title("2021-2022")

            plt.savefig(f"{docs_path}/Part_2/CD_plots/{id_key}_LCC.png")
            plt.close()
                
        except Exception as e:
            logging.error(f"Error in change detection for {id_key}: {e}")
            continue

def main():
    raster_list = load_raster(Path(f'{data_raw}/raster'))
    vector_list = load_vector(Path(f'{data_raw}/vector/Part_2'))
    clip_rasterDICT = batch_clip(raster_list, vector_list)
    LC_raster_plot(clip_rasterDICT)
    change_detection(clip_rasterDICT)

if __name__ == "__main__":
    main()