from dask import dataframe as dd


class Engine(object):
    def __init__(self) -> None:
        super().__init__()

        self.tables = dict()

    def register(self, name: str, data: dd.DataFrame):
        self.tables[name] = data

    def get_table(self, name: str) -> dd.DataFrame:
        return self.tables[name]
