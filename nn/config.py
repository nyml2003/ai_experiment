import time
import numpy as np
import pandas as pd

import utils

# 参数配置
learning_rate = 0.01  # 学习率
epochs = 1000  # 最大迭代次数
input_size = 5  # 输入层神经元个数
hidden_size = 10  # 隐藏层神经元个数
output_size = 3  # 输出层神经元个数
runs = 10  # 运行次数
activation = utils.sigmoid
activation_derivative = utils.sigmoid_derivative
loss_threshold = 0.7  # 损失阈值
# 生成随机种子
seed = int(time.time())
np.random.seed(seed)
seeds = np.random.randint(1000, 10000, runs)

def read_data():
    for filename in ['Iris-train.txt', 'Iris-test.txt']:
        print(f'读取数据集: {filename}')
        data: pd.DataFrame = pd.read_csv(
            filename,
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
        data: pd.DataFrame = data.sample(frac=1).reset_index(drop=True)
        x: np.ndarray = utils.normalize('min_max', data.iloc[:, :-1].values).T
        x: np.ndarray = np.insert(x, 0, -1, axis=0)
        y: np.ndarray = utils.one_hot_encode(data.iloc[:, -1].values, output_size).T
        yield x, y
