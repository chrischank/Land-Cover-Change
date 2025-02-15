##############################
#Part 2: LCC Batch Processing#
#Maintainer: Christopher Chan#
#Version: 0.1.6              #
#Date: 2025-02-14            #
##############################

import json
import logging
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import seaborn as sns
from osgeo import gdal
from rasterio.mask import mask

# path setup
BASE_PATH = Path(__file__).parent
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
    '''
    Load raster into gdal to get paths and info
    '''
    raster_list = []
    for path in glob(f'{raster_path}/*.tif'):
            ds = gdal.Open(str(path))
            raster_list.append(ds.GetDescription())

    return raster_list

def load_vector(vector_path: Path) -> List[tuple[gpd.GeoDataFrame, str]]:
    '''
    Load vector into a list of tuples
    '''
    vector_list = []
    for path in glob(f'{vector_path}/*.geojson'):
        ds = gpd.read_file(path)
        filename = Path(path).stem  # Gets filename without extension
        vector_list.append((ds, filename))
    return vector_list

def batch_clip(raster_list: List[str], vector_list: List[tuple[gpd.GeoDataFrame, str]]) -> dict:
    '''
    Batch processing for vector reproject and clipping,
    output as dict for east retrieval
    '''
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
    '''
    Plot Land Cover
    '''
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
    '''
    Massive function to calculate Change Detection
    Padding detection to max shape in set
    np.char for CD
    write and plot output
    '''
    out_dir = Path(f'{docs_path}/Part_2/CD_plots')
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir_result = Path(f'{data_model_output}/Part_2')
    out_dir_result.mkdir(parents=True, exist_ok=True)

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

            # Collect all the change detection into separate dataframes and output results
            result_df = pd.DataFrame({
                        'Land Cover 2020': pd.Series(raster_dict[id_key]["2020"]["raster"][0].flatten()).value_counts(),
                        'Land Cover 2021': pd.Series(raster_dict[id_key]["2021"]["raster"][0].flatten()).value_counts(),
                        'Land Cover 2022': pd.Series(raster_dict[id_key]["2022"]["raster"][0].flatten()).value_counts(),
                        #'LCC 2020-2021': pd.Series(LCC2020_2021_str.flatten()).value_counts(),
                        #'LCC 2021-2022': pd.Series(LCC2021_2022_str.flatten()).value_counts()
                        })
            LCC2020_2021_df = pd.Series(LCC2020_2021_str.flatten()).value_counts().reset_index().rename(columns={'index': 'LCC', 'count': 'LCC 2020-2021'})
            LCC2021_2022_df = pd.Series(LCC2021_2022_str.flatten()).value_counts().reset_index().rename(columns={'index': 'LCC', 'count': 'LCC 2021-2022'})

            result_df = result_df.reset_index().rename(columns={'index': 'LCC'})
            result_df["LCC"] = result_df["LCC"].astype(int).astype(str)

            result_df_LCC = pd.merge(result_df, LCC2020_2021_df, on='LCC', how='outer')
            result_df_LCC = pd.merge(result_df_LCC, LCC2021_2022_df, on='LCC', how='outer')

            def _map_cover_class(value, class_dict):
                '''
                Nested function map cover class from json
                originally README.txt
                '''
                # Check if the value is in the dictionary
                if value in class_dict:
                    return class_dict[value]
                # Check for combined values like '0:0' or 'NaN:NaN'
                elif ':' in str(value):
                    parts = str(value).split(':')
                    # Map each part and join them with a separator
                    return ':'.join(class_dict.get(part, 'Unknown') for part in parts)
                else:
                    return 'Unknown'

            # Load the class dictionary
            with open(f'{data_raw}/vector/LC_classes.json', encoding='utf-8-sig') as js:
                class_json = json.load(js)
                class_dict = class_json['classes']

            # Apply the mapping function only to result_df_LCC
            result_df_LCC['cover_class'] = result_df_LCC['LCC'].apply(lambda x: _map_cover_class(x, class_dict))

            # Calculation for area
            result_df_LCC["LC2020_m2"] = result_df_LCC["Land Cover 2020"]*10
            result_df_LCC["LC2021_m2"] = result_df_LCC["Land Cover 2021"]*10
            result_df_LCC["LC2022_m2"] = result_df_LCC["Land Cover 2022"]*10
            result_df_LCC["LCC2020_2021_m2"] = result_df_LCC["LCC 2020-2021"]*10
            result_df_LCC["LCC2021_2022_m2"] = result_df_LCC["LCC 2021-2022"]*10
            result_df_LCC.to_csv(f'{data_model_output}/Part_2/{id_key}_result_df.csv', index=False)

        except Exception as e:
            logging.error(f"Error in change detection for {id_key}: {e}")
            continue

def plot_LCC_area(id_key: str, result_df: pd.DataFrame) -> None:
    '''
    Scatterplot and Stackplot iteratively
    '''
    # Create output directory if it doesn't exist
    out_dir = Path(f'{docs_path}/Part_2/LCC_area_plots')
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(f'{data_raw}/vector/LC_classes.json', encoding='utf-8-sig') as js:
        class_json = json.load(js)
        class_dict = class_json['classes']

    # Ensure cover_class NaN values are mapped to 'Unknown'
    result_df['cover_class'] = result_df['cover_class'].fillna('Unknown')

    cols_to_plot = ["LC2020_m2", "LC2021_m2", "LC2022_m2"]

    # Reshape the dataframe from wide to long format
    melted_df = pd.melt(result_df, id_vars=['LCC', 'cover_class'],
                        value_vars=cols_to_plot,
                        var_name='category',
                        value_name='value')

    melted_df['value'] = pd.to_numeric(melted_df['value'], errors='coerce')

    # Replace NaN with 0 for plotting
    melted_df = melted_df.dropna(subset=['value'])

    # Plot scatterplot
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create scatter plot with fixed x-axis
    scatter = sns.scatterplot(data=melted_df, x='category', y='value', hue='cover_class', style='cover_class', s=100)

    # Set x-axis labels correctly
    plt.xticks(range(len(cols_to_plot)), ["Land Cover 2020", "Land Cover 2021", "Land Cover 2022"], rotation=45, ha="right")
    plt.ylabel("Area (m\u00B2)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"Land Cover 2020-2022 for {id_key}")
    plt.tight_layout()
    plt.savefig(f'{docs_path}/Part_2/LCC_area_plots/{id_key}LCC_Scatterplot.png')
    plt.close()

    # Clean data for stackplot
    valid_cover_classes = list(class_dict.values())
    result_df_Clean = result_df[result_df["cover_class"].isin(valid_cover_classes)]

    stack_data = {}
    for cover_class in result_df_Clean['cover_class'].unique():
        class_data = result_df_Clean[result_df_Clean['cover_class'] == cover_class][cols_to_plot].fillna(0).sum(axis=0)
        stack_data[cover_class] = class_data.values

    # Create the stackplot
    plt.figure(figsize=(12, 8))
    plt.stackplot(range(len(cols_to_plot)), *stack_data.values(), labels=stack_data.keys())
    plt.xticks(range(len(cols_to_plot)), ["Land Cover 2020", "Land Cover 2021", "Land Cover 2022"], rotation=45, ha="right")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f'Stackplot of Land Cover for {id_key}')
    plt.ylabel('Area (m\u00B2)')
    plt.tight_layout()
    plt.savefig(f'{docs_path}/Part_2/LCC_area_plots/{id_key}LCC_Stackplot.png')
    plt.close()

def main():
    raster_list = load_raster(Path(f'{data_raw}/raster'))
    vector_list = load_vector(Path(f'{data_raw}/vector/Part_2'))
    clip_rasterDICT = batch_clip(raster_list, vector_list)
    LC_raster_plot(clip_rasterDICT)
    change_detection(clip_rasterDICT)

    # Plot LCC area
    for df_path in sorted(glob(f'{data_model_output}/Part_2/*.csv')):
        # Extract ID from filename (e.g., "id_1_result_df.csv")
        filename = Path(df_path).stem  # gets 'id_1_result_df'
        id_num = filename.split('_')[1]  # gets '1'
        id_key = f"id_{id_num}"

        result_df = pd.read_csv(df_path, sep=',', na_values=[''], keep_default_na=True)
        plot_LCC_area(id_key, result_df)

    # Compute statistics for client
    all_df = glob(f'{data_model_output}/Part_2/*.csv')
    all_df_ls = [pd.read_csv(file) for file in all_df]
    concat_df = pd.concat(all_df_ls)
    final_df = concat_df.groupby("LCC", as_index=False).sum(numeric_only=True)

    # Remap LCC to cover_class
    with open(f'{data_raw}/vector/LC_classes.json', encoding='utf-8-sig') as js:
        class_json = json.load(js)
        class_dict = class_json['classes']

    def map_cover_class(value, class_dict):
        # Check if the value is in the dictionary
        if value in class_dict:
            return class_dict[value]
        # Check for combined values like '0:0' or 'NaN:NaN'
        elif ':' in value:
            parts = value.split(':')
            # Map each part and join them with a separator
            return ':'.join(class_dict.get(part, 'Unknown') for part in parts)
        else:
            return 'Unknown'

    final_df['cover_class'] = final_df['LCC'].apply(lambda x: map_cover_class(x, class_dict))

    out_dir = Path(f'{data_model_output}/Part_2/Concat')
    out_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(f'{data_model_output}/Part_2/Concat/final_df.csv', index=False)
    describe_df = final_df.describe()
    describe_df.to_csv(f'{data_model_output}/Part_2/Concat/final_df_describe.csv', index=True)

    # Stackplot output for combined dataframe
    valid_cover_classes = list(class_dict.values())
    final_df_clean = final_df[final_df["cover_class"].isin(valid_cover_classes)].drop(0, axis=0)

    cols_to_plot = ["LC2020_m2", "LC2021_m2", "LC2022_m2"]
    stack_data = {}

    out_dir = Path(f'{docs_path}/Part_2/Concat')
    out_dir.mkdir(parents=True, exist_ok=True)

    for cover_class in final_df_clean['cover_class'].unique():
        # Extract values for each cover_class and ensure NaNs are replaced with 0
        class_data = final_df_clean[final_df_clean['cover_class'] == cover_class][cols_to_plot].fillna(0).sum(axis=0)
        stack_data[cover_class] = class_data.values

    # Prepare x-axis labels
    x_labels = cols_to_plot

    # Create the stackplot
    plt.figure(figsize=(12, 8))
    plt.stackplot(x_labels, *stack_data.values(), labels=stack_data.keys())
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Stackplot of Combined Land Cover')
    plt.ylabel('Area (m\u00B2)')
    plt.xticks(range(len(cols_to_plot)), ["Land Cover 2020", "Land Cover 2021", "Land Cover 2022"], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f'{docs_path}/Part_2/Concat/LCC_Stackplot.png')
    plt.show()

if __name__ == "__main__":
    main()
