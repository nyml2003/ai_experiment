from collections import Counter
from math import log2

import pandas as pd


# 最后1列为结果，计算熵
def calculate_entropy(column: pd.Series) -> float:
    """
    计算熵
    :param column: 选中的列
    :return: 信息熵

    len_column是该列的长度

    Counter(column).values()是该列的每个值出现的次数

    count / len_column是该列每个值出现的概率

    log2(count / len_column)是该列每个值出现的概率的对数

    -(count / len_column) * log2(count / len_column)是该列每个值出现的概率的对数的相反数，即信息熵
    """
    len_column = len(column)
    return sum([
        -(count / len_column) * log2(count / len_column)
        for count in Counter(column).values()
    ])


# 条件熵


def gain_continuous(data: pd.DataFrame, column: int) -> tuple[float, float]:
    """
    计算连续值的信息增益
    :param data:数据集
    :param column:列
    :return: (信息增益，分割点)的元组

    返回信息增益是因为这只是一个列的最大信息增益，而不是整个数据集的信息增益，后续需要比较多个列的信息增益

    分割的方法

    本分割点取值为去重，排序后的列数据中两两相邻的值的中间值

    例如：1,2,3,4,5,6,7,8，分割点为1.5,2.5,3.5,4.5,5.5,6.5,7.5

    即最后选取的分割点是去重，排序后的列数据中的除了最小值的其他值

    按照这种分割方法，数据集每次会被分割成两部分，一部分是小于分割点的，一部分是大于等于分割点的

    分别计算条件熵，然后计算信息增益

    data[data[column] < chosen_spilt]是满足column列小于分割点的数据集的最后一列，用于计算条件熵

    信息增益=熵-条件熵，熵是固定的，需要最大的信息增益

    为了返回分割点，利用python的生成器表达式，返回一个元组，元组的第一个元素是信息增益，第二个元素是分割点

    取max时默认以元组的第一个元素为比较的值，即信息增益
    """
    entropy = calculate_entropy(data.iloc[:, -1])
    polished_data = data[column].drop_duplicates().sort_values()
    spilt_points = [(polished_data + polished_data.shift(-1)) / 2][0][1:-1]
    return max(
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
        for chosen_spilt in spilt_points
    )


def make_tree(data: pd.DataFrame, depth: int) -> dict | float:
    """
    生成决策树
    :param data: 数据集
    :param depth: 当前深度
    :return: 字典或者值

    预剪枝
    如果最后一列的值都相同，直接返回值

    如果深度大于5或者数据集的长度小于5，直接返回最后一列的众数,为了防止过拟合

    列的可能选择是除了最后一列的其他列，所以需要过滤掉最后一列

    如果该列的取值只有一个，那么这个列就没有必要再作为分割点了，所以需要过滤掉取值只有一个的列

    (
        gain_continuous(data, column),
        column
    )
    是一个元组，元组的第一个元素是（信息增益，分割点），第二个元素是列

    取max时默认以元组的第一个元素为比较的值，即（信息增益，分割点）

    比较（信息增益，分割点）的大小，实际先比较信息增益的大小

    因为信息增益并不是要输出的信息，我们只需要分割点和对应的列
    """
    if data.iloc[:, -1].nunique() == 1:
        return data.iloc[0, -1]
    if depth > 2 or data.shape[0] < 5:
        return Counter(data.iloc[:, -1]).most_common(1)[0][0]

    ((_, spilt), column) = max([
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
    return {f"{column}小于{spilt}": make_tree(data[data[column] < spilt], depth + 1),
            f"{column}大于等于{spilt}": make_tree(data[data[column] >= spilt], depth + 1)}
