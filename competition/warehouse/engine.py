from typing import Union

import pandas as pd
import cfgrib
import xarray as xa
import geopandas as gpd


class Engine(object):
    def __init__(self) -> None:
        super().__init__()
        self.tables = dict()

    def register_table(self,
                       name: str,
                       path: str) -> None:
        if path.endswith('.csv'):
            self.tables[name] = pd.read_csv(path)
        elif path.endswith('.grib'):
            self.tables[name] = xa.merge([xd.reset_coords()[list(xd.data_vars)]
                                          for xd in cfgrib.open_datasets(path)])
        elif path.endswith('.geojson'):
            self.tables[name] = gpd.read_file(path)
        elif path.endswith('.pbf'):
            raise NotImplementedError
        else:
            raise NotImplementedError

    def get_table(self, name: str) -> Union[pd.DataFrame, xa.Dataset, gpd.GeoDataFrame]:
        return self.tables[name]
