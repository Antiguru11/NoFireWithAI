import pandas as pd

from .base import CalcerBase
from ..warehouse import Engine
from ..repository import Repository


class Featuriser():
    def __init__(self,
                 engine: Engine, 
                 repository: Repository, ) -> None:
        self.engine = engine
        self.repository = repository

    def create_calcer(self, name: str, args: dict = {}) -> CalcerBase:
        args['engine'] = self.engine

        return self.repository.get_object(name)(**args)

    
    def get_features(self,
                     config: dict,)-> pd.DataFrame:
        keys = ['dt', 'grid_index']
        dataframes = list()   
        for name, args in config.items():
            calcer = self.create_calcer(name, args)
            dataframes.append(calcer.compute())

        features_df = dataframes[0]
        for df in dataframes[1:]:
            features_df = features_df.merge(df, how='outer', on=keys)

        return features_df