from abc import ABC, abstractmethod

import numpy as np
from dask import dataframe as dd

from ..warehouse import Engine


class CalcerBase(ABC):
    name = None
    keys = None

    def __init__(self, engine: Engine, sample_table: str) -> None:
        super().__init__()
        self.engine = engine
        self.sample_table = sample_table

    @abstractmethod
    def compute(self) -> dd.DataFrame:
        pass

    def set_grid_index(self,
                       data: dd.DataFrame, 
                       lon_col: str,
                       lat_col: str, 
                       lon_min: float = 19.0,
                       lon_max: float = 169.0,
                       lat_min: float = 41.0,
                       lat_max: float = 81.5,
                       step: float = 0.2, ) -> None:
        # latitudes = np.arange(lat_min, lat_max, step).round(1)
        longitudes = np.arange(lon_min, lon_max, step).round(1)

        col_n = len(longitudes)
        col_i = ((data[lon_col] - lon_min) / step).astype(int)
        row_i = ((data[lat_col] - lat_min) / step).astype(int)

        data['grid_index'] = (row_i * col_n + col_i).astype(int)


class SampleCalcer(CalcerBase):
    name = 'sample'
    keys = ['dt', 'grid_index', ]

    def compute(self) -> dd.DataFrame:
        return self.engine.get_table(self.sample_table)[self.keys]


class DatesFeaturesCalcer(CalcerBase):
    name = 'dates_features'
    keys = ['dt', 'grid_index', ]

    def compute(self) -> dd.DataFrame:
        sample_dd = self.engine.get_table(self.sample_table)

        features_dd = sample_dd[self.keys]
        dt = dd.to_datetime(sample_dd['dt']).dt
        features_dd['year'] = dt.year
        features_dd['month'] = dt.month
        features_dd['week'] = dt.isocalendar().week
        features_dd['day_of_week'] = dt.dayofweek

        return features_dd


class TargetBaseCalcer(CalcerBase): 
    name = 'target_base'
    keys = ['dt', 'grid_index', ]

    def compute(self) -> dd.DataFrame:
        sample_dd = self.engine.get_table(self.sample_table)

        infires_col = [f'infire_day_{i}' for i in range(1, 9)]
        features_dd = sample_dd[self.keys + infires_col]

        for i in range(8, 0, -1):
            features_dd['infire_day_num'] = features_dd[f'infire_day_{i}'].where(features_dd[f'infire_day_{i}'] != 1, i)
        features_dd['infire_day_num'] = features_dd['infire_day_num'].fillna(0)

        return features_dd

