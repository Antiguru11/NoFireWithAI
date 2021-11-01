from abc import ABC, abstractmethod

import pandas as pd
from sklearn.pipeline import Pipeline as _Pipeline

from ..repository import Repository


class PipelineBase(ABC):
    def __init__(self,
                 repository: Repository,
                 config: dict, ) -> None:
        super().__init__()
        self.repository = repository
        self.config = config

    @abstractmethod
    def fit(self, features_df: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        pass    

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    def get_pipeline(self, **kwargs) -> _Pipeline:
        steps = list()
        for group in self.config['estimators'].keys():
            steps += [(ename, self.repository.get_object(ename)(eargs)) 
                      for ename, eargs in self.config['estimators'][group].items()]
        return _Pipeline(steps, **kwargs)

    def get_splitter(self):
        raise NotImplementedError

    def get_searcher(self):
        raise NotImplementedError

    def fit_pipeline(self):
        raise NotImplementedError
