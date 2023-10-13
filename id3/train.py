from collections import Counter
from math import log2

import pandas as pd


# 最后1列为结果，计算熵
def calculate_entropy(column: pd.Series) -> float:
    return sum([
        -(count / len(column)) * log2(count / len(column))
        for count in Counter(column).values()
    ])


# 条件熵


def gain_continuous(data: pd.DataFrame, column: int) -> tuple[float, float]:
    """
    计算连续值的信息增益
    :param data:数据集
    :param column:列
    :return: (信息增益，分割点)
    """
    # 去重，排序
    # 假设是5,6,7,8，那么分割点是<5.5,<6.5,<7.5，如果我使用<6,<7,<8是等价的
    # 1,2,3,4 可以被划分为
    # 1 | 2,3,4 分割点：<2
    # 1,2 | 3,4 分割点：<3
    # 1,2,3 | 4 分割点：<4
    # 信息增益=熵-条件熵，熵是固定的，信息增益最大
    entropy = calculate_entropy(data.iloc[:, -1])
    return max([
        (
            entropy
            -
            len(data[data[column] < chosen_spilt]) / len(data) * calculate_entropy(
                data[data[column] < chosen_spilt].iloc[:, -1]
            )
            -
            len(data[data[column] >= chosen_spilt]) / len(data) * calculate_entropy(
                data[data[column] >= chosen_spilt].iloc[:, -1]
            ),
            chosen_spilt
        )
        for chosen_spilt in data[column].drop_duplicates().sort_values()[1:]
    ])


def make_tree(data: pd.DataFrame, depth: int) -> dict | float:
    if data.iloc[:, -1].nunique() == 1:
        return data.iloc[0, -1]
    if depth > 5:
        return Counter(data.iloc[:, -1]).most_common(1)[0][0]
    if data.shape[0] < 5:
        return Counter(data.iloc[:, -1]).most_common(1)[0][0]

    ((max_gain, spilt), column) = max([
        (
            gain_continuous(data, column),
            column
        )
        for column in filter(
            lambda col:
            col != data.columns[-1]
            and
            data[col].nunique() > 1,
            data.columns
        )
    ])
    if max_gain == 1:
        return Counter(data.iloc[:, -1]).most_common(1)[0][0]
    else:
        return {f"{column}小于{spilt}": make_tree(data[data[column] < spilt], depth + 1),
                f"{column}大于等于{spilt}": make_tree(data[data[column] >= spilt], depth + 1)}
