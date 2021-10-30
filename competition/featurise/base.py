from abc import ABC, abstractmethod

import pandas as pd
import geopandas as gpd

from ..warehouse import Engine
from ..utils import set_grid_index


class CalcerBase(ABC):
    name = None
    keys = ['dt', 'grid_index', ]

    def __init__(self, engine: Engine) -> None:
        super().__init__()
        self.engine = engine

    @abstractmethod
    def compute(self) -> pd.DataFrame:
        pass


class GeoBaseCalcer(CalcerBase):
    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)

    def prepare_data(self) -> pd.DataFrame:
        geo_df = self.engine.get_table('geo')

        geo_df = geo_df.loc[:, ['population', 'place', 'geometry']]
        geo_df.loc[:, 'city_lon'] = geo_df['geometry'].x
        geo_df.loc[:, 'city_lat'] = geo_df['geometry'].y
        geo_df.loc[geo_df['city_lon'] < 0, 'city_lon'] += 360
        
        mask = geo_df['population'].notna()
        geo_df.loc[mask, 'population'] = (geo_df
                                          .loc[mask, 'population']
                                          .fillna('')
                                          .apply(lambda x: x.split('(')[0])
                                          .str.replace(" ", "")
                                          .astype(int))
        
        set_grid_index(geo_df, lon_col='city_lon', lat_col='city_lat')

        geo_df = (geo_df
                  .loc[geo_df['place'] != 'city_block', :]
                  .sort_values('population', ascending=False)
                  .drop_duplicates(subset=['grid_index'], keep='first')
                  .drop(columns=['geometry'])
                  .reset_index(drop=True))

        return geo_df
