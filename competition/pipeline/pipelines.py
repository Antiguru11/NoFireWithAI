import os
import logging

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import f1_score, precision_score

from .base import PipelineBase
from ..repository import Repository


class BaselinePipeline(PipelineBase):
    name = 'baseline_pipeline'

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.models = dict.fromkeys(['infire_day_num'] 
                                     + [f'infire_day_{i}' for i in range(1, 9, 1)])
        self.features = None
        # self.cat_features = ['month',
        #                      'week',
        #                      'day_of_week', 'geo_place' ]

    def fit(self, features_df: pd.DataFrame, config: dict, repository: Repository):
        logging.info(f'Fit pipeline')
        self.config = config
        self.repository = repository

        split_dt = pd.to_datetime(self.config['split_dt'])

        targets = list(self.models.keys())
        self.features = list(set(features_df.columns.tolist()) 
                             - set(targets) 
                             - set(['dt', 'grid_index']))

        features_df = features_df.sort_values(by='dt', ascending=True)

        targets_df = features_df.loc[:, targets[1:]]
        targets_df = (targets_df
                      .replace(0, np.nan)
                      .fillna(axis=1, method="ffill")
                      .fillna(0)
                      .astype(int))
        targets_df.loc[:, targets[0]] = features_df[targets[0]]


        for i, target in enumerate(targets):
            logging.info(f'Fit - {target}')
            multiclass = target == 'infire_day_num'

            X = features_df.loc[:, self.features]
            y = targets_df.loc[:, target]

            split_mask = pd.to_datetime(features_df['dt']) < split_dt

            (X_train, X_test,
             y_train, y_test,) = (X.loc[split_mask, :], X.loc[~split_mask, :],
                                  y[split_mask], y[~split_mask],)
            
            pipeline = self.get_pipeline(config['estimators'])
            pipeline.set_params(model__eval_metric='MultiClass' if multiclass else 'F1',
                                model__random_seed=self.seed+i)
            # pipeline.set_params(model__cat_features=[i 
            #                                          for i, f in enumerate(self.features) 
            #                                          if f in self.cat_features], )
            logging.info(f'Read pipeline - {pipeline}')

            spliter = self.get_splitter(config['split'])
            logging.info(f'Read spliter - {spliter}')

            searcher = self.get_searcher(config['search'])
            if searcher is not None:
                searcher.scoring = 'f1_micro' if multiclass else 'f1'
            logging.info(f'Read searcher - {searcher}')

            fit_params = config['fit_params']

            pipeline = self.fit_pipeline(X_train,
                                         y_train,
                                         pipeline,
                                         spliter,
                                         searcher,
                                         fit_params)
            
            quality = f1_score(y_test, pipeline.predict(X_test),
                               average='micro' if multiclass else 'binary')

            logging.info(f'Pipeline quality - {quality}')

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
