import os
import time
import pandas as pd
import numpy as np
import utils

# 参数配置
learning_rate = 0.01  # 学习率
epochs = 500  # 最大迭代次数
input_size = 4  # 输入层神经元个数
hidden_size = 10  # 隐藏层神经元个数
output_size = 3  # 输出层神经元个数
runs = 10  # 运行次数
activation = utils.sigmoid
activation_derivative = utils.sigmoid_derivative
normalize = utils.normalize('min_max')
init_weight = 0.01  # 权重初始化范围 [-init_weight, init_weight]
loss_threshold = 0.07  # 损失阈值
# 生成随机种子
seed = int(time.time())
np.random.seed(seed)
seeds = np.random.randint(0, 10000, runs)

# 数据预处理
data_path = 'data'


def read_data_config():
    for filename in ['Iris-train.txt', 'Iris-test.txt']:
        data: pd.DataFrame = utils.read_data(os.path.join(data_path, filename))
        data: pd.DataFrame = data.sample(frac=1)
        x: np.ndarray = utils.normalize('min_max', data.iloc[:, :-1].values).T
        x: np.ndarray = np.insert(x, 0, -1, axis=0)
        y: np.ndarray = utils.one_hot_encode(data.iloc[:, -1].values, output_size).T
        yield x, y


train_x, train_y = next(read_data_config())
test_x, test_y = next(read_data_config())
input_size += 1
