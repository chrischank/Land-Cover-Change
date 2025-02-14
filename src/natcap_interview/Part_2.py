##############################
#Part 2: LCC Batch Processing#
#Maintainer: Christopher Chan#
#Version: 0.0.1              #
#Date: 2025-02-14            #
##############################

import os
import json

import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from PIL import Image
from osgeo import gdal
from rasterio.mask import mask
from rasterio.enums import Resampling


# path setup
BASE_PATH = Path(os.getcwd())
data_raw = (BASE_PATH/'../../data/01_raw').resolve()
data_intermediate = (BASE_PATH/'../../data/02_intermediate').resolve()
data_model_output = (BASE_PATH/'../../data/07_model_output').resolve()
docs_path = (BASE_PATH/'../../docs').resolve()

# load data
def load_raster(raster_path: Path) -> list:
    raster_list = []
    for path in raster_path.glob('*.tif'):
        print(path)
        try:
            ds = gdal.Open(str(path))
            if ds is None:
                print(f"Failed to open {path}")
                continue
            raster_list.append(ds)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    return raster_list

def main():
    raster_list = load_raster(Path(f'{data_raw}/raster'))
    print(raster_list)

if __name__ == "__main__":
    main()