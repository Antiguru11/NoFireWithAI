import os
import gc
from typing import List

import cfgrib
import pandas as pd

from .engine import Engine


class Warehouse:
    def __init__(self,
                 rawdata_path: str,
                 dwhdata_path: str,):
        self.rawdata_path = rawdata_path
        self.dwhdata_path = dwhdata_path
        self.paths_used = list()

        if not os.path.exists(self.dwhdata_path):
            Warehouse.execute_cmd(f'mkdir {self.dwhdata_path}')

    @staticmethod
    def execute_cmd(cmd: str) -> None:
        if os.system(cmd) != 0:
            raise OSError

    def create(self,
               use_geo: bool = True,
               use_era5: bool = True, 
               era5_metrics: List[str] = None,
               era5_years: List[int] = None) -> Engine:
        engine = Engine()

        for name, filename in [('train', 'train.csv'),
                               ('train_raw', 'train_raw.csv'),
                               ('sample_test', 'sample_test.csv'), ]:
            self.append(engine, name, filename)

        if use_geo:
            self.append(engine, 'city_town_village', 'city_town_village.geojson')
            self.append(engine, 'russia_latest_osm', 'russia-latest.osm.pbf')
        
        if use_era5:
            for name in era5_metrics:
                for year in era5_years:
                    self.append(engine,
                                f'{name}_{year}',
                                'ERA5_data/' + f'{name}_{year}.grib')

        return engine

    def append(self, engine: Engine, name: str, filename: str):
        rawfile_path = os.path.join(self.rawdata_path, filename)
        dwhfile_path = os.path.join(self.dwhdata_path, 
                                    f'{name}.' + filename.split('.')[-1])

        path = dwhfile_path
        if not os.path.exists(dwhfile_path):
            Warehouse.execute_cmd(f"cp {rawfile_path} {dwhfile_path}")

        if not path.endswith('.csv'):
            csvfile_path = os.path.join(self.dwhdata_path, name + '.csv')
            Warehouse.to_csv(dwhfile_path, csvfile_path)
            Warehouse.execute_cmd(f"rm {dwhfile_path}")

            path = csvfile_path
        
        engine.register_table(name, path)
        self.paths_used.append(path)

    @staticmethod
    def to_csv(source_path: str, destination_path: str) -> None:
        if source_path.endswith('.geojson'):
            Warehouse.geojson_to_csv(source_path, destination_path)
        elif source_path.endswith('.pbf'):
            Warehouse.pbf_to_csv(source_path, destination_path)
        elif source_path.endswith('.grib'):
            Warehouse.grib_to_csv(source_path, destination_path)
        else:
            pass

    @staticmethod
    def geojson_to_csv(source_path, destination_path):
        raise NotImplementedError

    @staticmethod
    def pbf_to_csv(source_path, destination_path):
        raise NotImplementedError

    @staticmethod
    def grib_to_csv(source_path: str, destination_path: str) -> None:
        xdatasets = cfgrib.open_datasets(source_path)

        dataframes = list()
        for i in range(len(xdatasets)):
            dataframes.append((xdatasets[i]
                               .reset_coords()[list(xdatasets[i].data_vars)]
                               .to_dataframe()))
            xdatasets[i] = None
            gc.collect()
        
        dataframe = pd.concat(dataframes).reset_index()
        dataframe.to_csv(destination_path)
        del dataframe
        


