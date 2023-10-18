import os
import pandas as pd
import numpy as np

import util

# 参数配置
learning_rate = 0.01  # 学习率
epochs = 10000  # 最大迭代次数
input_size = 4
hidden_size = 10
output_size = 3
seed = 114
runs = 10
activation = util.sigmoid
activation_derivative = util.sigmoid_derivative
normalize = util.normalize('min_max')

# 数据预处理
data_path = 'data'


def read_data_config():
    for filename in ['Iris-train.txt', 'Iris-test.txt']:
        data: pd.DataFrame = util.read_data(os.path.join(data_path, filename))
        x: np.ndarray = util.normalize('min_max', data.iloc[:, :-1].values).T
        y: np.ndarray = util.one_hot_encode(data.iloc[:, -1].values, output_size).T
        yield x, y


train_x, train_y = next(read_data_config())
train_x = np.insert(train_x, 0, -1, axis=0)
test_x, test_y = next(read_data_config())
test_x = np.insert(test_x, 0, -1, axis=0)
input_size += 1
