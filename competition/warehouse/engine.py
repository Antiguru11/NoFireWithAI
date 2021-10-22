import os
import logging

from dask import dataframe as dd


class Engine(object):
    def __init__(self) -> None:
        super().__init__()
        self.tables = dict()

    def register_table(self,
                       name: str,
                       path: str) -> None:
        logging.info(f'register {name}')
        self.tables[name] = dd.read_csv(path)

    def get_table(self, name: str) -> dd.DataFrame:
        return self.tables[name]
