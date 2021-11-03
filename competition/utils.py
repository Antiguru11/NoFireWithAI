import re
from typing import Union, Tuple

import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import xarray as xa
import geopandas as gpd

era5_all_metrics = ['evaporation1', 'evaporation2',
                    'heat1', 'heat2',
                    'temp', 'vegetation', 'wind', ]
era5_all_years = [2018, 2019, 2020, 2021, ]


grib_lon_min = 19.0
grib_lon_max = 169.0
grib_lat_min = 41.0
grib_lat_max = 81.5


grid_step = 0.2
grid_n_rows = int(np.ceil((grib_lat_max - grib_lat_min) / grid_step))
grid_n_columns = int(np.ceil((grib_lon_max - grib_lon_min) / grid_step))


def set_grid_index(data: Union[pd.DataFrame, xa.Dataset, gpd.GeoDataFrame], 
                   lon_col: str,
                   lat_col: str, ) -> None:
    longitudes = np.arange(grib_lon_min, grib_lon_max, grid_step).round(1)

    col_n = len(longitudes)
    col_i = ((data[lon_col] - grib_lon_min) / grid_step).astype(int)
    row_i = ((data[lat_col] - grib_lat_min) / grid_step).astype(int)

    data['grid_index'] = (row_i * col_n + col_i).astype(int)


def make_pooling(array: np.ndarray,
                 kernel_size: int,
                 func: callable) -> np.ndarray:
    input = np.full(shape=grid_n_rows * grid_n_columns,
                    fill_value=np.nan,
                    dtype=array.dtype)

    min_len = min(len(input), len(array))
    input[:min_len] = array[:min_len]

    shape = (grid_n_rows, grid_n_columns)
    padded = np.pad(input.reshape(shape),
                    pad_width=tuple([kernel_size // 2 for _ in shape]),
                    constant_values=np.nan)
    
    item_size = array.strides[0]
    pooling_shape = (shape[0], shape[1],
                     kernel_size, kernel_size)
    pooling_strides = (padded.shape[1] * item_size, item_size, 
                       padded.shape[1] * item_size, item_size, )
    polling = as_strided(padded, shape=pooling_shape, strides=pooling_strides)

    mask = (~np.isnan(polling)).any(axis=(2,3))
    result = np.full(input.shape, np.nan, dtype=input.dtype)
    result[mask.reshape(-1)] = func(polling[mask], axis=(1,2)).reshape(-1)
    
    return result


def camel2snake(name: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
