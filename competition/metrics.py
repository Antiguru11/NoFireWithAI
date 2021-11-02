import numpy as np
import pandas as pd


def competition_metric(y_true: pd.DataFrame,
                       y_pred: pd.DataFrame,
                       fail_coef: float = 2,
                       C: float = 5):
    assert y_true.shape == y_pred.shape
    assert all([el in [0, 1] for el in np.unique(y_pred.values)])

    y_true_sums = (y_true
                   .replace(0, np.nan)
                   .fillna(axis=1, method="ffill")
                   .fillna(0)
                   .sum(axis=1))
    
    y_pred_sums = (y_pred
                   .replace(0, np.nan)
                   .fillna(axis=1, method="ffill")
                   .fillna(0)
                   .sum(axis=1))
    
    max_penalty = y_true.shape[1] * fail_coef

    days_error_series = y_pred_sums - y_true_sums
    days_error_series.loc[days_error_series < 0] = days_error_series[days_error_series < 0] * (-fail_coef)

    metric = (days_error_series
              .apply(lambda x: (C ** (x / max_penalty) - 1) / (C - 1))
              .mean())
    return round(1 - metric, 5)
