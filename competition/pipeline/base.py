from abc import ABC, abstractclassmethod, abstractmethod

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as _Pipeline

from ..repository import Repository


class PipelineBase(ABC):
    name = None

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.seed = seed

    @abstractmethod
    def fit(self,
            features_df: pd.DataFrame,
            config: dict,
            repository: Repository):
        pass

    @abstractmethod
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        pass    

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractclassmethod
    def load(path: str) -> None:
        pass

    def get_pipeline(self, config: dict, **kwargs) -> _Pipeline:
        steps = list()

        transformers = list()
        for tconfig in config['transformers']:
            tname, tcolumns, testimators = list(tconfig.values())

            tsteps = list()
            for estimator in testimators:
                ename, eargs = list(estimator.items())[0]
                tsteps.append((ename, self.repository.get_object(ename)(**eargs),))
            
            transformers.append((tname, _Pipeline(tsteps), tcolumns))

        steps.append(('transformers', ColumnTransformer(transformers, remainder='passthrough')))
        
        for selector in config['selectors']:
            sname, sargs = list(selector.items())[0]
            steps.append((sname, self.repository.get_object(sname)(**sargs)))
        
        mname, margs = list(config['model'].items())[0]
        steps.append(('model', self.repository.get_object(mname)(**margs)))

        return _Pipeline(steps, **kwargs)

    def get_splitter(self, config: dict):
        if config:
            return self.repository.get_object(config['name'])(**config['args'])
        return None

    def get_searcher(self, config: dict):
        if config:
            return self.repository.get_object(config['name'])(estimator=None,
                                                              **config['args'])
        return None

    def fit_pipeline(self,
                     X: pd.DataFrame,
                     y: pd.Series,
                     pipeline: _Pipeline,
                     spliter=None,
                     searcher=None, 
                     fit_params: dict = {}, ) -> _Pipeline:
        best_params = pipeline.get_params(deep=True)
        if searcher is not None:
            searcher.estimator = pipeline
            searcher.cv = spliter
            searcher.fit(X, y, **fit_params)
            best_params.update(searcher.best_params_)

        pipeline.set_params(**best_params)
        pipeline.fit(X, y, **fit_params)

        return pipeline
