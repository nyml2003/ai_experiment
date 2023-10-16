import pandas as pd


def read_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep='\t', header=None, skiprows=[0], skipfooter=1, engine='python')


def calc_accuracy(tree: dict, data: pd.DataFrame) -> float:
    return sum(
        [
            1
            for i in range(len(data))
            if predict(tree, data.iloc[i, :]) == data.iloc[i, -1]
        ]
    ) / len(data)


# 后剪枝
def post_pruning(tree: dict):
    if isinstance(tree, dict):
        key = list(tree.keys())[0]
        index, label = key.split('小于')
        if isinstance(tree[f'{index}小于{label}'], dict) or isinstance(tree[f'{index}大于等于{label}'], dict):
            return {
                f'{index}小于{label}': post_pruning(tree[f'{index}小于{label}']),
                f'{index}大于等于{label}': post_pruning(tree[f'{index}大于等于{label}'])
            }
        else:
            if tree[f'{index}小于{label}'] == tree[f'{index}大于等于{label}']:
                return tree[f'{index}小于{label}']
            else:
                return {
                    f'{index}小于{label}': tree[f'{index}小于{label}'],
                    f'{index}大于等于{label}': tree[f'{index}大于等于{label}']
                }
    else:
        return tree
    

def predict(tree: dict, data: pd.Series):
    if isinstance(tree, dict):
        key = list(tree.keys())[0]
        index, label = key.split('小于')
        if data[int(index)] < float(label):
            return predict(tree[f'{index}小于{label}'], data)
        else:
            return predict(tree[f'{index}大于等于{label}'], data)
    else:
        return tree





