import gc
from abc import ABC, abstractmethod
from typing import List, Union
from datetime import timedelta

import numpy as np
import pandas as pd

from ..warehouse import Engine
from ..utils import (set_grid_index,
                     make_pooling, )


class CalcerBase(ABC):
    name = None
    keys = ['dt', 'grid_index', ]

    def __init__(self, engine: Engine) -> None:
        super().__init__()
        self.engine = engine

    @abstractmethod
    def compute(self) -> pd.DataFrame:
        pass


class DatesFeaturesCalcer(CalcerBase):
    name = 'dates_features'

    def compute(self) -> pd.DataFrame:
        features_df = self.engine.get_table('sample').loc[:, self.keys]

        dt = pd.to_datetime(features_df['dt']).dt
        features_df.loc[:, 'month'] = dt.month
        features_df.loc[:, 'week'] = dt.isocalendar().week.astype(int)
        features_df.loc[:, 'day_of_week'] = dt.dayofweek

        return features_df


class GeoFeaturesCalcer(CalcerBase):
    name = 'geo_features'

    def compute(self) -> pd.DataFrame:
        features_df = self.engine.get_table('geo')

        return features_df


class GribFeaturesCalcer(CalcerBase):
    name = 'grib_features'
    ufunc_map = {'max': np.nanmax,
                 'min': np.nanmin,
                 'mean': np.nanmean, }

    def __init__(self, engine: Engine,
                 metric: str, 
                 window_size: int,
                 lags: List[int],
                 agg_funcs: List[str]) -> None:
        super().__init__(engine)

        self.metric = metric
        self.window_size = window_size
        self.lags = lags
        self.agg_funcs = agg_funcs

    def compute(self) -> pd.DataFrame:
        sample_df = self.engine.get_table('sample').loc[:, self.keys]
        
        years = list(pd.to_datetime(sample_df['dt']).dt.year.unique())
        dt_min = pd.to_datetime(sample_df['dt']).min() - pd.DateOffset(max(self.lags))
        dt_max = pd.to_datetime(sample_df['dt']).max() - pd.DateOffset(min(self.lags))
        years = list(sorted(set(years + [dt_min.year, dt_max.year])))

        dates = pd.to_datetime(sample_df['dt']).dt.date.unique()
        for lag in self.lags:
            dates = np.concatenate([dates, dates - timedelta(days=lag)])
        dates = list(np.unique(dates))
        
        grid_idxs = list(sample_df['grid_index'].unique())

        features_df = sample_df.copy()

        parts = list()
        for year in years:
            grib_ds = (self.engine
                       .get_table(f'{self.metric}_{year}'))

            set_grid_index(grib_ds,
                           lon_col='longitude', 
                           lat_col='latitude', )

            for dt, group_ds in grib_ds.groupby(grib_ds['time'].dt.date):
                if not dt in dates:
                    continue

                agg_df = (group_ds
                          .to_dataframe()
                          .groupby('grid_index')[list(grib_ds.keys())[:-1]].mean())

                for column in agg_df.columns:
                    for func in self.agg_funcs:
                        name = f'{self.metric}_{column}_{func}'
                        agg_df.loc[:, name] = make_pooling(agg_df[column].values,
                                                           self.window_size,
                                                           self.ufunc_map[func])

                agg_df = (agg_df
                          .loc[(agg_df.notna().any(axis=1) 
                                & agg_df.index.isin(grid_idxs)), :]
                          .reset_index()
                          .assign(dt=str(dt)))
                    
                parts.append(agg_df)

        metric_df = pd.concat(parts)
        columns = list(set(metric_df.columns) - set(self.keys))

        del parts
        gc.collect()

        for lag in self.lags:
            features_df['dt'] = ((pd.to_datetime(sample_df['dt']) - pd.DateOffset(lag))
                                .dt.strftime('%Y-%m-%d'))
                
            features_df = (features_df
                           .merge(metric_df.rename(columns={c: f'{c}_lag_{lag}' 
                                                            for c in columns}),
                                  how='left',
                                  on=['dt', 'grid_index'], ))

        del metric_df 
        gc.collect()
        
        features_df.loc[:, 'dt'] = sample_df['dt']
        return features_df


class TargetBaseCalcer(CalcerBase): 
    name = 'target_base'

    def compute(self) -> pd.DataFrame:
        sample_df = self.engine.get_table('sample')

        infires_col = [f'infire_day_{i}' for i in range(1, 9)]
        target_df = sample_df.loc[:, self.keys + infires_col]

        for i in range(8, 0, -1):
            target_df.loc[:, 'infire_day_num'] = target_df[f'infire_day_{i}'].where(target_df[f'infire_day_{i}'] != 1, i)
        target_df.loc[:, 'infire_day_num'] = target_df['infire_day_num'].fillna(0)

        return target_df

