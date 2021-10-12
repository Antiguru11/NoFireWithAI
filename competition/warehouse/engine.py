import logging
from typing import Union

import geopandas as gpd
import cfgrib
import pandas as pd
from dask import dataframe as dd


class Engine(object):
    def __init__(self) -> None:
        super().__init__()
        self.tables = dict()

    def register_table(self,
                       name: str,
                       path_or_data: Union[str, dd.DataFrame, pd.DataFrame]) -> None:
        logging.info(f'register {name}')
        if type(path_or_data) == str:
            try:
                if path_or_data.endswith('.csv'):
                    self.tables[name] = dd.read_csv(path_or_data)
                elif path_or_data.endswith('.geojson'):
                    self.tables[name] = dd.from_pandas(gpd.read_file(path_or_data),
                                                    npartitions=1, )
                elif path_or_data.endswith('.pbf'):
                    raise NotImplementedError
                elif path_or_data.endswith('.grib'):
                    xarrays = cfgrib.open_datasets(path_or_data)
                    dataframe = pd.concat([xa.to_dataframe().reset_index() 
                                           for xa in xarrays],
                                           axis=1, )
                    self.tables[name] = dd.from_pandas(dataframe,
                                                    npartitions=1, )
                else:
                    raise ValueError
            except Exception as ex:
                logging.info(f'error register {name}: {ex}')
        else:
            if isinstance(path_or_data, pd.DataFrame):
                self.tables[name] = dd.from_pandas(path_or_data,
                                                   npartitions=1, )
            elif isinstance(path_or_data, dd.DataFrame):
                self.tables[name] = path_or_data
            else:
                raise ValueError


    def get_table(self, name: str) -> dd.DataFrame:
        return self.tables[name]
