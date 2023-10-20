import numpy as np
from tqdm import trange

import config
from train import NeuralNetwork

if __name__ == '__main__':
    # 记录每次运行的准确率
    res_arr = []

    for run in range(config.runs):
        print()
        print(f'第{run + 1}次运行,随机种子: {config.seeds[run]}')
        read_data_gen = config.read_data()
        train_x, train_y = next(read_data_gen)
        test_x, test_y = next(read_data_gen)
        # 用时间戳作为随机种子
        np.random.seed(config.seeds[run])
        nn = NeuralNetwork(
            inputs=train_x,
            outputs=train_y,
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
        )
        with trange(config.epochs) as t:
            for epoch in t:
                t.set_description(f'Epoch {epoch + 1}')
                nn.train(config.learning_rate)
                if epoch % 1000 == 0:
                    loss = nn.loss()
                    t.set_postfix(loss=loss)
                    if loss < config.loss_threshold:
                        break
        accuracy_train = nn.evaluate(train_x, train_y)
        accuracy_test = nn.evaluate(test_x, test_y)
        print(f'训练准确率: {accuracy_train}')
        print(f'测试准确率: {accuracy_test}')
        res_arr.append(accuracy_test)
        print()
    print(f'平均准确率: {np.mean(res_arr)}')
    print(f'标准差: {np.std(res_arr)}')
