import sys
import inspect
from typing import Dict

import pandas as pd

from .calcers import *
from ..warehouse import Engine


class Featuriser():
    _calcers = dict()

    @staticmethod
    def register_calcers():
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if name.endswith('Calcer') and name != 'CalcerBase':
                Featuriser._calcers[obj.name] = obj
                

    def __init__(self,
                 engine: Engine, ) -> None:
        Featuriser.register_calcers()

        self.engine = engine

    def create_calcer(self, name: str, args: Dict[str, object] = {}) -> CalcerBase:
        args['engine'] = self.engine

        return self._calcers[name](**args)

    
    def get_features(self,
                     config: Dict[str, Dict[str, object]],)-> pd.DataFrame:
        keys = ['dt', 'grid_index']
        dataframes = list()   
        for name, args in config.items():
            calcer = self.create_calcer(name, args)
            dataframes.append(calcer.compute())

        features_df = dataframes[0]
        for df in dataframes[1:]:
            features_df = features_df.merge(df, how='outer', on=keys)

        return features_df
