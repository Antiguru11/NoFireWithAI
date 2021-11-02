import os
import logging
from typing import List, Union

from .engine import Engine
from ..utils import era5_all_metrics, era5_all_years


class Warehouse:
    def __init__(self,
                 data_path: str,
                 train=True, ):
        self.data_path = data_path
        self.train = train

    def create(self,
               use_geo: bool = True,
               use_era5: bool = True, 
               era5_metrics: Union[str, List[str]] = '*',
               era5_years: Union[str, List[str]] = '*') -> Engine:
        engine = Engine()

        if self.train:
            self.append(engine, 'sample', 'train.csv')
        else:
            self.append(engine, 'sample', 'test.csv')

        if use_geo:
            self.append(engine, 'geo', 'city_town_village.geojson')
        
        if use_era5:
            metrics = era5_all_metrics if era5_metrics == '*' else era5_metrics
            years = era5_all_years if era5_years == '*' else era5_years
            for name in metrics:
                for year in years:
                    self.append(engine,
                                f'{name}_{year}',
                                'ERA5_data/' + f'{name}_{year}.grib')

        return engine

    def append(self, engine: Engine, name: str, filename: str):
        logging.info(f'Read data - {name}')
        path = os.path.join(self.data_path, filename)        
        engine.register_table(name, path)

