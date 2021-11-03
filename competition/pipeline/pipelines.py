import os
import logging

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from .base import PipelineBase
from ..repository import Repository


class BaselinePipeline(PipelineBase):
    name = 'baseline_pipeline'

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.models = dict.fromkeys(['infire_day_num'] 
                                     + [f'infire_day_{i}' for i in range(1, 9, 1)])
        self.features = None

    def fit(self, features_df: pd.DataFrame, config: dict, repository: Repository):
        logging.info(f'Fit pipeline')
        self.config = config
        self.repository = repository

        targets = list(self.models.keys())
        self.features = list(set(features_df.columns.tolist()) 
                             - set(targets) 
                             - set(['dt', 'grid_index']))

        targets_df = features_df.loc[:, targets[1:]]
        targets_df = (targets_df
                      .replace(0, np.nan)
                      .fillna(axis=1, method="ffill")
                      .fillna(0)
                      .astype(int))
        targets_df.loc[:, targets[0]] = features_df[targets[0]]


        for i, target in enumerate(targets):
            logging.info(f'Fit - {target}')

            X = features_df.loc[:, self.features]
            y = targets_df.loc[:, target]

            (X_train, X_test,
             y_train, y_test,) = train_test_split(X, y,
                                                  test_size=0.3,
                                                  stratify=y,
                                                  random_state=self.seed)
            
            pipeline = self.get_pipeline(config['estimators'], verbose=True)
            pipeline.set_params(model__eval_metric='MultiClass' if target == 'infire_day_num' else 'F1',
                                model__random_seed=self.seed+i)

            spliter = self.get_splitter(config['split'])
            searcher = self.get_searcher(config['search'])
            fit_params = config['fit_params']

            pipeline = self.fit_pipeline(X_train,
                                         y_train,
                                         pipeline,
                                         spliter,
                                         searcher,
                                         fit_params)
            quality = f1_score(y_test, pipeline.predict(X_test),
                               average='micro' if target == 'infire_day_num' else 'binary', )

            self.models[target] = pipeline

        return self

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f'Pipeline predict')

        X = features_df.loc[:, self.features]

        predicts_df = pd.DataFrame()
        for i in range(1, 9):
            name = f'infire_day_{i}'
            predicts_df.loc[:, name] = (self.models[name].predict_proba(X)[:, 1] > 0.51).astype(int)

        name = 'infire_day_num'
        predicts_df.loc[:, name] = self.models[name].predict(X)

        mask = predicts_df[[f'infire_day_{i}' for i in range(1, 9)]].sum(axis=1) == 0
        for i in range(1, 9):
            name = f'infire_day_{i}'
            predicts_df.loc[mask & (predicts_df['infire_day_num'] == i), name] = 1

        predicts_df = predicts_df.drop(columns=['infire_day_num'])
        return predicts_df

    def save(self, path: str) -> None:
        logging.info(f'Save pipeline')

        with open(os.path.join(path, 'features.pkl'), 'wb') as file:
            pickle.dump(self.features, file)

        for name, model in self.models.items():
            with open(os.path.join(path, name + '.pkl'), 'wb') as file:
                pickle.dump(model, file)

    def load(path: str, seed: int):
        logging.info(f'Load pipeline')
        pipeline = BaselinePipeline(seed)

        with open(os.path.join(path, 'features.pkl'), 'rb') as file:
            pipeline.features = pickle.load(file)

        for name in pipeline.models:
            with open(os.path.join(path, name + '.pkl'), 'rb') as file:
                pipeline.models[name] = pickle.load(file)
        return pipeline
