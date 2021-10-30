import gc
from typing import List, Union
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from .base import CalcerBase, GeoBaseCalcer
from ..warehouse import Engine
from ..utils import (grid_n_rows,
                     grid_n_columns,
                     set_grid_index,
                     make_pooling, )


class DatesFeaturesCalcer(CalcerBase):
    name = 'dates_features'

    def compute(self) -> pd.DataFrame:
        features_df = self.engine.get_table('sample').loc[:, self.keys]

        dt = pd.to_datetime(features_df['dt']).dt
        features_df.loc[:, 'month'] = dt.month
        features_df.loc[:, 'week'] = dt.isocalendar().week.astype(int)
        features_df.loc[:, 'day_of_week'] = dt.dayofweek

        return features_df


class GeoCatFeaturesCalcer(GeoBaseCalcer):
    name = 'geo_cat_features'

    def compute(self) -> pd.DataFrame:
        sample_df = self.engine.get_table('sample')
        geo_df = self.prepare_data()

        features_df = (sample_df
                       .loc[:, self.keys]
                       .merge(geo_df,
                              how='left',
                              on='grid_index'))
        features_df = (features_df
                       .drop(columns=['city_lon', 'city_lat'])
                       .rename(columns={c: f'geo_{c}' 
                                        for c in features_df.columns[2:]}))

        return features_df

class GeoNeighborsFeaturesCalcer(GeoBaseCalcer):
    name = 'geo_neighbors_features'

    def __init__(self, engine: Engine, count_neighbors: int) -> None:
        super().__init__(engine)
        self.count_neighbors = count_neighbors

    def compute(self) -> pd.DataFrame:
        sample_df = self.engine.get_table('sample')
        geo_df = self.prepare_data()

        tree = KDTree(geo_df[['city_lon', 'city_lat']])
        metrics = list()
        for _, row in sample_df.iterrows():
            lon_center = (row['lon_min'] + row['lon_max']) / 2
            lat_center = (row['lat_min'] + row['lat_max']) / 2
            dist, _ = tree.query([[lon_center, lat_center]],
                                 k=self.count_neighbors)
            if len(dist) > 0:
                metrics.append((np.mean(dist),
                                np.max(dist),
                                np.min(dist), ))
            else:
                metrics.append((np.nan, ) * 3)
        features_df = pd.DataFrame(metrics,
                                   columns=['city_mean_distance',
                                            'city_max_distance',
                                            'city_min_distance', ],
                                   index=sample_df.index)
                                   
        features_df = (sample_df
                       .loc[:, self.keys]
                       .join(features_df,
                             how='left'))
        features_df = (features_df
                       .rename(columns={c: f'geo_cn{self.count_neighbors}_{c}' 
                                        for c in features_df.columns[2:]}))

        return features_df


class GribFeaturesCalcer(CalcerBase):
    name = 'grib_features'
    ufunc_map = {'max': np.nanmax,
                 'min': np.nanmin,
                 'mean': np.nanmean, }

    def __init__(self, engine: Engine,
                 metric: str, 
                 pooling_size: int,
                 lags: List[int],
                 agg_funcs: List[str]) -> None:
        super().__init__(engine)

        self.metric = metric
        self.pooling_size = pooling_size
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
        all_grid_idxs_df = pd.DataFrame(index=pd.RangeIndex(stop=grid_n_rows * grid_n_columns,
                                                            name='grid_index'))

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

                group_df = (group_ds
                            .to_dataframe()
                            .groupby('grid_index')[list(grib_ds.keys())[:-1]].mean())
                group_df = all_grid_idxs_df.join(group_df, how='left')

                for column in group_df.columns:
                    for func in self.agg_funcs:
                        name = f'{self.metric}_{column}_ws{self.pooling_size}_{func}'
                        group_df.loc[:, name] = make_pooling(group_df[column].values,
                                                             self.pooling_size,
                                                             self.ufunc_map[func])
                    del group_df[column]

                group_df = (group_df
                            .loc[(group_df.notna().any(axis=1) 
                                  & group_df.index.isin(grid_idxs)), :]
                            .reset_index()
                            .assign(dt=str(dt)))
                    
                parts.append(group_df)

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

