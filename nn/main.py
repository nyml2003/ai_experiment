import time

import numpy as np

import config
from train import NeuralNetwork

if __name__ == '__main__':
    res_arr=[]
    for i in range(config.runs):
        # 用时间戳作为随机种子
        seed = int(time.time())
        nn = NeuralNetwork(
            config.train_x,
            config.train_y,
            config.input_size,
            config.hidden_size,
            config.output_size,
            seed,
            config.learning_rate
        )
        print(f'第{i + 1}次运行,随机种子: {seed}')
        nn.train(config.epochs)
        accuracy_train = nn.evaluate(config.train_x, config.train_y)
        accuracy_test = nn.evaluate(config.test_x, config.test_y)
        print(f'训练准确率: {accuracy_train}')
        print(f'测试准确率: {accuracy_test}')
        res_arr.append(accuracy_test)
    print(f'平均准确率: {np.mean(res_arr)}')
    print(f'标准差: {np.std(res_arr)}')
