import pandas as pd
from typing import List


def read_data(file_path: str):
    return pd.read_csv(file_path, sep='\t', header=None, skiprows=[0], skipfooter=1, engine='python').values.tolist()


def calc_accuracy(tree, data):
    accuracy = 0
    for i in data:
        if predict(tree, i) == i[-1]:
            accuracy += 1
        else:
            pass
    return accuracy / len(data)


def predict(tree: dict, data: List[float]):
    if isinstance(tree, dict):
        key = list(tree.keys())[0]
        index, label = key.split('<')
        if data[int(index)] < float(label):
            return predict(tree[f'{index}<{label}'], data)
        else:
            return predict(tree[f'{index}>{label}'], data)
    else:
        return tree
