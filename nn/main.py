import numpy as np
from tqdm import trange

import config
from train import NeuralNetwork

if __name__ == '__main__':
    # 读取数据
    train_x = config.train_x
    train_y = config.train_y
    test_x = config.test_x
    test_y = config.test_y

    # 记录每次运行的准确率
    res_arr = []

    for run in range(config.runs):
        print()
        print(f'第{run + 1}次运行,随机种子: {config.seeds[run]}')
        # 用时间戳作为随机种子
        np.random.seed(config.seeds[run])
        nn = NeuralNetwork(
            inputs=train_x,
            outputs=train_y,
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            learning_rate=config.learning_rate,
            init_weight=config.init_weight
        )
        print(f'开始训练,最大迭代次数: {config.epochs}')
        for epoch in trange(config.epochs):
            nn.train()
            if epoch > config.epochs / 2 and nn.loss() < 0.05:
                break
        print(f'训练完成,迭代次数: {epoch + 1},损失: {nn.loss()}')
        accuracy_train = nn.evaluate(config.train_x, config.train_y)
        accuracy_test = nn.evaluate(config.test_x, config.test_y)
        print(f'训练准确率: {accuracy_train}')
        print(f'测试准确率: {accuracy_test}')
        res_arr.append(accuracy_test)
        print()
    print(f'平均准确率: {np.mean(res_arr)}')
    print(f'标准差: {np.std(res_arr)}')
