import numpy as np
import config
import util


class Neuron:

    def __init__(self, from_size: int, to_size: int, init_weight=0.01):
        """
        初始化权重和偏置
        :param from_size:
        :param to_size:

        两种情况： input-->hidden  hidden-->output

        """
        self.weights = np.random.rand(to_size, from_size) * 2 - 1

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

    def __init__(self, inputs, outputs, input_size, hidden_size, output_size, seed, learning_rate):
        np.random.seed(seed)
        self.inputs = inputs
        self.outputs = outputs
        self.data_size = inputs.shape[1]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layer = Neuron(input_size, hidden_size)
        self.output_layer = Neuron(hidden_size, output_size)
        self.learning_rate = learning_rate

    def loss(self):
        hidden_output = self.hidden_layer.forward(self.inputs)
        output_output = self.output_layer.forward(hidden_output)
        return np.sum(np.square(self.outputs - output_output)) / self.data_size

    def train(self, epochs):
        loss_arr = []
        for epoch in range(epochs):
            # forward
            hidden_output = self.hidden_layer.forward(self.inputs)
            output_output = self.output_layer.forward(hidden_output)
            # backward
            output_delta = self.output_layer.backward(output_output - self.outputs, output_output)
            hidden_delta = self.hidden_layer.backward(self.output_layer.weights.T @ output_delta, hidden_output)
            # update
            self.output_layer.weights -= self.learning_rate * (output_delta @ hidden_output.T)
            self.hidden_layer.weights -= self.learning_rate * (hidden_delta @ self.inputs.T)
            if epoch % 1000 == 0:
                loss = self.loss()
                loss_arr.append({
                    'epoch': epoch,
                    'loss': loss
                })
                print(f'epoch: {epoch}, loss: {self.loss()}')
                if loss < 0.01:
                    break
        return loss_arr

    def predict(self, inputs):
        hidden_output = self.hidden_layer.forward(inputs)
        output_output = self.output_layer.forward(hidden_output)
        return output_output

    def evaluate(self, inputs, outputs):
        return np.sum(np.argmax(self.predict(inputs), axis=0) == np.argmax(outputs, axis=0)) / self.data_size
