from .base import BaseSelector

import pandas as pd


class DummySelector(BaseSelector):
    def fit(self, X, y=None, **fit_params):
        self.mask = (pd.DataFrame(X).nunique() > 1).values
        return self

    def _more_tags(self):
        return {'allow_nan': True, }
