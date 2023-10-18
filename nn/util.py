import pandas as pd
import numpy as np


def read_data(file_path: str) -> pd.DataFrame:
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


# def normalize(data: np.ndarray, method: str) -> np.ndarray:
#     def wrapper(data: np.ndarray) -> np.ndarray:
#         return normalize(data, method)
#     match method:
#         case 'min_max':
#             return (data - np.min(data)) / (np.max(data) - np.min(data))
#         case 'z_score':
#             return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
#         case _:
#             raise ValueError('type must be min_max or z_score')

def normalize(method: str, _data: np.ndarray = None) -> np.ndarray:
    def wrapper(data: np.ndarray) -> np.ndarray:
        match method:
            case 'min_max':
                return (data - np.min(data)) / (np.max(data) - np.min(data))
            case 'z_score':
                return (data - np.mean(data)) / np.std(data)
            case _:
                raise ValueError('type must be min_max or z_score')
    if _data is None:
        return wrapper
    else:
        return wrapper(_data)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


def one_hot_encode(data_output: np.ndarray, output_size) -> np.ndarray:
    return np.eye(output_size)[data_output]
