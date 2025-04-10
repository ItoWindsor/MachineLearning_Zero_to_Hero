import numpy as np
import pandas as pd
from typing import Union


def custom_weighted_accuracy(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
) -> float:
    correct_direction = (np.sign(y_true) == np.sign(y_pred)).astype(int)
    score = np.sum(correct_direction * np.abs(y_true), axis=0) / np.sum(np.abs(y_true), axis=0)

    return score


def concatenate_train_and_test(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
) -> pd.DataFrame:
    X_train_temp = X_train.copy()
    X_train_temp['set'] = 'train'
    X_test_temp = X_test.copy()
    X_test_temp['set'] = 'test'

    df = pd.concat([X_train_temp, X_test_temp], axis=0)
    df.set_index('DELIVERY_START', inplace=True, drop=True)

    return df


def separate_train_and_test(
        df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train_updated = df[df.set == 'train'].copy(deep=True)
    X_test_updated = df[df.set == 'test'].copy(deep=True)

    X_train_updated.drop(['set'], axis=1, inplace=True)
    X_test_updated.drop(['set'], axis=1, inplace=True)

    X_train_updated.reset_index(inplace=True, drop=True)
    X_test_updated.reset_index(inplace=True, drop=True)

    return X_train_updated, X_test_updated
