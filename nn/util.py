import pandas as pd
import numpy as np


def read_data(file_path: str) -> np.ndarray:
    return pd.read_csv(
        file_path,
        sep=' ',
        index_col=False,
        names=[
            'sepal_length',
            'sepal_width',
            'petal_length',
            'petal_width',
            'class'
        ]
    )


def normalize(df: pd.DataFrame, method: str) -> pd.DataFrame:
    match method:
        case 'min_max':
            return (df - df.min()) / (df.max() - df.min())
        case 'z_score':
            return (df - df.mean()) / df.std()
        case _:
            raise ValueError('type must be min_max or z_score')


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


def one_hot_encode(y: pd.Series) -> pd.DataFrame:
    return pd.get_dummies(y)
