from typing import Dict

import pandas as pd

from .calcers import *
from ..warehouse import Engine


class Featuriser():
    _calcers = {c.name: c for c in [SampleCalcer,
                                    DatesFeaturesCalcer,
                                    TargetBaseCalcer]}

    def __init__(self,
                 engine: Engine,
                 sample_table: str,
                 target_name: str = None,
                 target_args: Dict[str, object] = {}) -> None:
        self.engine = engine
        self.sample_table = sample_table
        self.sample_calcer = self.create_calcer('sample')
        self.target_calcer = None
        if target_name is not None:
            self.target_calcer = self.create_calcer(target_name,
                                                    target_args, )

    def create_calcer(self, name: str, args: Dict[str, object] = {}) -> CalcerBase:
        args['engine'] = self.engine
        args['sample_table'] = self.sample_table

        return self._calcers[name](**args)

    
    def get_features(self,
                     config: Dict[str, Dict[str, object]],)-> pd.DataFrame:
        keys = self.sample_calcer.keys
        features_dd = self.sample_calcer.compute()           
        for name, args in config.items():
            calcer = self.create_calcer(name, args)
            features_dd = features_dd.merge(calcer.compute(),
                                            how='left',
                                            on=keys)

        if self.target_calcer is not None:
            features_dd = features_dd.merge(self.target_calcer.compute(),
                                            how='inner',
                                            on=keys)

        return features_dd.compute()
