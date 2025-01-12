import numpy as np
import pandas as pd
from typing import Union


def custom_weighted_accuracy(y_true: Union[pd.Series,np.ndarray], y_pred: Union[pd.Series,np.ndarray]) -> float:
    correct_direction = (np.sign(y_true) == np.sign(y_pred))
    return np.sum(correct_direction * np.abs(y_true), axis=0) / np.sum(np.abs(y_true), axis=0)
