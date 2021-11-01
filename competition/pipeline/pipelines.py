import pandas as pd

from .base import PipelineBase


class BaselinePipeline(PipelineBase):
    name = 'baseline_pipeline'

    def fit(self, features_df: pd.DataFrame):
        raise NotImplementedError

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
