import numpy as np
import config


class Neuron:

    def __init__(self, from_size: int, to_size: int):
        """
        初始化权重和偏置
        :param from_size:
        :param to_size:

        两种情况：
        - 从输入层到隐藏层：input-->hidden
        - 从隐藏层到输出层：hidden-->output
        神经元的连接权重一般采用 [−1, 1] 区间，均值为 0 的随机数。
        """
        self.weights = np.random.rand(to_size, from_size) * 2 - 1

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        前向传播
        :param inputs:
        :return:

        因为在输入层中加入了-1，所以这里不需要再加入偏置
        """
        return self.activation(np.dot(self.weights, inputs))

    @staticmethod
    def activation(inputs: np.ndarray) -> np.ndarray:
        return config.activation(inputs)

    @staticmethod
    def activation_derivative(inputs: np.ndarray) -> np.ndarray:
        return config.activation_derivative(inputs)

    def update(self, gradient: np.ndarray, learning_rate):
        self.weights -= learning_rate * gradient


class NeuralNetwork:

    def __init__(self, inputs, outputs, input_size, hidden_size, output_size):
        self.inputs = inputs
        self.outputs = outputs
        # 数据集大小
        self.data_size = inputs.shape[1]
        self.hidden_layer = Neuron(input_size, hidden_size)
        self.output_layer = Neuron(hidden_size, output_size)

    def forward(self, inputs):
        hidden_output = self.hidden_layer.forward(inputs)
        output_output = self.output_layer.forward(hidden_output)
        return hidden_output, output_output

    def backward(self, hidden_output, output_output):
        output_error = output_output - self.outputs
        output_gradient = output_error @ hidden_output.T

        hidden_error = self.output_layer.weights.T @ output_error * Neuron.activation_derivative(hidden_output)
        hidden_gradient = hidden_error @ self.inputs.T

        return hidden_gradient, output_gradient

    def predict(self, inputs):
        hidden_output = self.hidden_layer.forward(inputs)
        predict_outputs = self.output_layer.forward(hidden_output)
        return predict_outputs

    def evaluate(self, inputs, outputs):
        """
        :param inputs:
        :param outputs:
        :return:

        比较每个列向量的最大值的索引是否相等

        axis = 0 表示按列比较
        """
        predict_outputs = np.argmax(self.predict(inputs), axis=0)
        outputs = np.argmax(outputs, axis=0)
        return np.sum(predict_outputs == outputs) / self.data_size

    def loss(self):
        """
        :param outputs:
        :return:

        均方误差损失函数
        """
        predict_outputs = self.predict(self.inputs)
        return np.sum(
            np.square(
                np.linalg.norm(predict_outputs - self.outputs, axis=0)
            )
        ) / self.data_size
