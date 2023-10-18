import numpy as np
import config


class Neuron:

    def __init__(self, from_size: int, to_size: int, init_weight: int) -> None:
        """
        初始化权重和偏置
        :param from_size:
        :param to_size:

        两种情况：
        - 从输入层到隐藏层：input-->hidden
        - 从隐藏层到输出层：hidden-->output
        权重初始化范围 [-init_weight, init_weight]
        """
        self.weights = np.random.uniform(-init_weight, init_weight, (to_size, from_size))

    @staticmethod
    def activation(x: np.ndarray) -> np.ndarray:
        return config.activation(x)

    @staticmethod
    def activation_derivative(x: np.ndarray) -> np.ndarray:
        return config.activation_derivative(x)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.activation(np.dot(self.weights, inputs))

    def backward(self, diff: np.ndarray, output: np.ndarray) -> np.ndarray:
        return np.multiply(diff, self.activation_derivative(output))


class NeuralNetwork:

    def __init__(self, inputs, outputs, input_size, hidden_size, output_size, init_weight):
        self.inputs = inputs
        self.outputs = outputs
        # 数据集大小
        self.data_size = inputs.shape[1]
        self.input_size = input_size
        self.hidden_layer = Neuron(input_size, hidden_size, init_weight)
        self.output_layer = Neuron(hidden_size, output_size, init_weight)

    def loss(self):
        _, output_output = self.forward(self.inputs)
        return np.sum(np.square(self.outputs - output_output)) / self.data_size

    def forward(self, inputs):
        hidden_output = self.hidden_layer.forward(inputs)
        output_output = self.output_layer.forward(hidden_output)
        return hidden_output, output_output

    def backward(self, hidden_output, output_output):
        output_delta = self.output_layer.backward(
            output_output - self.outputs,
            output_output
        )
        hidden_delta = self.hidden_layer.backward(
            np.dot(self.output_layer.weights.T, output_delta),
            hidden_output
        )
        return output_delta, hidden_delta

    def update(self, output_delta, hidden_delta, learning_rate):
        self.output_layer.weights -= learning_rate * np.dot(output_delta, self.hidden_layer.forward(self.inputs).T)
        self.hidden_layer.weights -= learning_rate * np.dot(hidden_delta, self.inputs.T)

    def train(self, learning_rate):
        hidden_output, output_output = self.forward(self.inputs)
        output_delta, hidden_delta = self.backward(hidden_output, output_output)
        self.update(output_delta, hidden_delta, learning_rate)

    def predict(self, inputs):
        _, output_output = self.forward(inputs)
        return output_output

    def evaluate(self, inputs, outputs):
        return np.sum(np.argmax(self.predict(inputs), axis=0) == np.argmax(outputs, axis=0)) / self.data_size
