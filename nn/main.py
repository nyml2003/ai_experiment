from time import sleep

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from nn import utils
from train import NeuralNetwork
import config


def read_data(output_size=config.output_size):
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
        x: np.ndarray = utils.normalize('z_score', data.iloc[:, :-1].values).T
        x: np.ndarray = np.insert(x, 0, -1, axis=0)
        y: np.ndarray = utils.one_hot_encode(data.iloc[:, -1].values, output_size).T
        yield x, y


if __name__ == '__main__':
    # 记录每次运行的准确率
    arr = []

    for run in range(config.runs):
        print()
        print(f'第{run + 1}次运行,随机种子: {config.seeds[run]}')
        read_data_gen = read_data()
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
        last_loss = None
        learning_rate = config.learning_rate
        for epoch in range(config.epochs):
            # 前向传播：
            # - 隐藏层：$a^{(1)} = g(z^{(1)})$, $z^{(1)} = W^{(1)}a^{(0)}}$
            # - 输出层：$a^{(2)} = g(z^{(2)})$, $z^{(2)} = W^{(2)}a^{(1)}}$
            # 反向传播：
            # - 输出层误差：$\delta^{(2)} = a^{(2)} - y$
            # - 隐藏层误差：$\delta^{(1)} = (W^{(2)})^T\delta^{(2)} \odot g'(z^{(1)})$
            # - 权重梯度：$\frac{\partial J}{\partial W^{(l)}} = \delta^{(l+1)} (a^{(l)})^T$
            # 更新权重：使用梯度下降法更新权重
            # - $W^{(l)} = W^{(l)} - \eta \frac{\partial J}{\partial W^{(l)}}$
            # 前向传播
            hidden_output, output_output = nn.forward(train_x)
            # 反向传播
            hidden_gradient, output_gradient = nn.backward(hidden_output, output_output)
            # 更新权重
            nn.hidden_layer.update(hidden_gradient, learning_rate)
            nn.output_layer.update(output_gradient, learning_rate)
            # 计算损失函数,根据损失函数的值调整学习率
            loss = nn.loss()
            if last_loss is not None:
                if abs(loss - last_loss) < 0.00001:
                    print(f'损失函数变化小于0.00001, 停止训练')
                    break
                if loss > last_loss:
                    learning_rate *= 0.9
                    if learning_rate < 0.00001:
                        learning_rate = 0.00001
                elif loss < last_loss:
                    learning_rate *= 1.1
                    if learning_rate > 0.01:
                        learning_rate = 0.01
            last_loss = loss
            if epoch % (config.epochs // 10) == 0:
                print(f'训练次数: {epoch}, 损失: {loss}, 学习率: {learning_rate}')
        print(f'训练次数: {epoch}, 损失: {loss}')
        accuracy_train = nn.evaluate(train_x, train_y)
        accuracy_test = nn.evaluate(test_x, test_y)
        print(f'训练准确率: {accuracy_train}')
        print(f'测试准确率: {accuracy_test}')
        arr.append(accuracy_test)
        print()
    print(f'平均准确率: {np.mean(arr)}')
    print(f'标准差: {np.std(arr)}')
