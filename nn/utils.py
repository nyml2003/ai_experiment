import numpy as np


def normalize(method: str, data: np.ndarray):
    """
    归一化
    :param method: 选择归一化方式
    :param data: 数据集
    :return:

    ppt中介绍的两种归一化方式：
    - min_max
    用每一维度 j 的最小值 min_j和最大值 max_j进行规范化
    - z_score
    用每一维度 j 的均值 mean_j和标准差 std_j进行规范化
    """
    match method:
        case 'min_max':
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
            return (data - data_min) / (data_max - data_min)
        case 'z_score':
            data_std = np.std(data, axis=0)
            data_mean = np.mean(data, axis=0)
            return (data - data_mean) / data_std
        case _:
            raise ValueError('type must be min_max or z_score')


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    sigmoid函数
    :param x: 矩阵
    :return: sigmoid函数的输出

    对于矩阵x的每一个元素，计算sigmoid函数的输出
    使用numpy数组时，通常使用numpy函数而不是运算符来执行逐元素操作。这是因为numpy函数可以更好地处理广播和其他特殊情况。
    本函数即逐元素操作
    """
    return np.divide(1, np.add(1, np.exp(-x)))


def sigmoid_derivative(fx: np.ndarray) -> np.ndarray:
    """
    sigmoid函数的导数
    :param fx: sigmoid函数的输出
    :return: sigmoid函数的导数

    在前向传播中，sigmoid函数的输出为f(x)
    在反向传播中，sigmoid函数的导数为f(x) * (1 - f(x))
    但是没有必要再计算一次sigmoid函数，因此在前向传播时，将sigmoid函数的输出保存下来，以便在反向传播时直接使用
    这里的fx就是sigmoid函数的输出
    仅保留了根据fx计算导数的函数
    """
    return np.multiply(fx, 1 - fx)


def one_hot_encode(y: np.ndarray, output_size) -> np.ndarray:
    """
    one-hot编码
    :param y: 输出层的真实值
    :param output_size: 输出层神经元个数
    :return: one-hot编码后的矩阵

    np.eye()生成对角矩阵

    y是1维数组，每个元素是一个类别的索引
    """
    return np.eye(output_size)[y]
