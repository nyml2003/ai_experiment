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
        """
        self.weights = np.random.rand(to_size, from_size) * 2 - 1

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.activation(np.dot(self.weights, inputs))

    @staticmethod
    def activation(inputs: np.ndarray) -> np.ndarray:
        return config.activation(inputs)

    def update(self, new_weights: np.ndarray):
        self.weights = new_weights


class NeuralNetwork:

    def __init__(self, inputs, outputs, input_size, hidden_size, output_size):
        self.inputs = inputs
        self.outputs = outputs
        # 数据集大小
        self.data_size = inputs.shape[1]
        self.hidden_layer = Neuron(input_size, hidden_size)
        self.output_layer = Neuron(hidden_size, output_size)

    def train(self, learning_rate):
        """
        训练1次
        :param learning_rate: 学习率
        :return:

        这里权重需要多次使用，所以先保存下来

        前向传播：
        - 隐藏层的输出：hidden_output = sigmoid(hidden_weights * inputs)
        - 输出层的输出：output_output = sigmoid(output_weights * hidden_output)

        反向传播：

        """
        # 权重
        hidden_weights = self.hidden_layer.weights
        output_weights = self.output_layer.weights
        # 前向传播
        hidden_output = config.activation(
            np.dot(hidden_weights, self.inputs)
        )
        output_output = config.activation(
            np.dot(output_weights, hidden_output)
        )
        # 反向传播
        output_delta = np.multiply(
            np.subtract(output_output, self.outputs),
            config.activation_derivative(
                output_output
            )
        )
        hidden_delta = np.multiply(
            np.dot(output_weights.T, output_delta),
            config.activation_derivative(
                hidden_output
            )
        )
        # 更新权重
        self.output_layer.update(
            np.subtract(
                output_weights,
                np.multiply(
                    np.dot(
                        output_delta,
                        hidden_output.T
                    ),
                    learning_rate
                )
            )
        )
        self.hidden_layer.update(
            np.subtract(
                hidden_weights,
                np.multiply(
                    np.dot(
                        hidden_delta,
                        self.inputs.T
                    ),
                    learning_rate
                )
            )
        )

    def predict(self, inputs):
        hidden_output = self.hidden_layer.forward(inputs)
        return self.output_layer.forward(hidden_output)

    def evaluate(self, inputs, outputs):
        predict_outputs = np.argmax(self.predict(inputs), axis=0)
        outputs = np.argmax(outputs, axis=0)
        return np.sum(predict_outputs == outputs) / self.data_size

    def loss(self):
        # 1/2 * ()**2
        output = self.predict(self.inputs)
        return np.sum(np.power(np.subtract(output, self.outputs), 2)) / 2
