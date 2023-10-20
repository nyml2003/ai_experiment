import time

import numpy as np
from tqdm import trange

import config
from train import NeuralNetwork

if __name__ == '__main__':
    # 记录每次运行的准确率
    res_arr = []
    np.random.seed(int(time.time()))
    for run in range(config.runs):
        nn = NeuralNetwork(
            inputs=config.train_x,
            outputs=config.train_y,
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
        )
        print(f'开始训练,最大迭代次数: {config.epochs}')
        for epoch in range(config.epochs):
            nn.train(config.learning_rate)
        print(f'训练完成,迭代次数: {epoch + 1}, 损失: {nn.losses()}')
        accuracy_train = nn.evaluate(config.train_x, config.train_y)
        accuracy_test = nn.evaluate(config.test_x, config.test_y)
        print(f'训练准确率: {accuracy_train}')
        print(f'测试准确率: {accuracy_test}')
        res_arr.append(accuracy_test)
        print()
    print(f'平均准确率: {np.mean(res_arr)}')
    print(f'标准差: {np.std(res_arr)}')
