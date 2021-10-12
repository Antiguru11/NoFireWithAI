from abc import ABC, abstractmethod

from dask import dataframe as dd

from ..warehouse import Engine


class CalcerBase(ABC):
    name = None
    keys = None

    def __init__(self, engine: Engine, sample_table: str) -> None:
        super().__init__()
        self.engine = engine

    @abstractmethod
    def compute(self) -> dd.DataFrame:
        pass


class DateFeaturesCalcer(CalcerBase):
    name = 'date_features'
    keys = ['grid_index', ]

    def compute(self) -> dd.DataFrame:
        raise NotImplementedError



