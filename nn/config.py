import time
import numpy as np

import utils

# 参数配置
# 学习率 η 的设置为 (0, 1] 的区间 0.00001 ~ 0.01
learning_rate = 0.01
# 训练次数一般大于 500 epochs。
epochs = 1000
# 训练样本, 其中，x
# [−1, x1, x2, x3, x4], 所以输入层神经元个数为 4 + 1 = 5
input_size = 5
hidden_size = 10  # 隐藏层神经元个数
output_size = 3  # 输出层神经元个数
runs = 10  # 运行次数

activation = utils.sigmoid
activation_derivative = utils.sigmoid_derivative
# 生成随机种子
seed = int(time.time())
np.random.seed(seed)
seeds = np.random.randint(1000, 10000, runs)



