import numpy as np
import config


class Neuron:
    activation = np.vectorize(config.activation)
    activation_derivative = np.vectorize(config.activation_derivative)

    def __init__(self, from_size, to_size):
        self.weights = np.random.randn(from_size, to_size)
        self.bias = np.zeros((1, to_size))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.activation(np.dot(inputs, self.weights) + self.bias)


class NeuralNetwork:

    def __init__(self, inputs, outputs, input_size, hidden_size, output_size, seed, learning_rate):
        np.random.seed(seed)
        self.inputs = inputs
        self.outputs = outputs
        self.data_size = len(inputs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layer = Neuron(input_size, hidden_size)
        self.output_layer = Neuron(hidden_size, output_size)
        self.learning_rate = learning_rate

    def forward(self):
        return self.output_layer.forward(self.hidden_layer.forward(self.inputs))

    def backward(self, output):
        output_error = output - self.outputs
        output_delta = output_error * self.output_layer.activation_derivative(output)
        hidden_error = np.dot(output_delta, self.output_layer.weights.T)
        hidden_delta = hidden_error * self.hidden_layer.activation_derivative(self.hidden_layer.forward(self.inputs))
        self.output_layer.weights -= self.learning_rate * np.dot(self.hidden_layer.forward(self.inputs).T, output_delta)
        self.output_layer.bias -= self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.hidden_layer.weights -= self.learning_rate * np.dot(self.inputs.T, hidden_delta)
        self.hidden_layer.bias -= self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def loss(self):
        return np.sum(
            np.power(self.outputs - self.forward(), 2)
        ) / len(self.outputs)

    def train(self, epochs):
        loss_arr = []
        for epoch in range(epochs):
            output = self.forward()
            self.backward(output)
            if epoch % 1000 == 0:
                loss = self.loss()
                loss_arr.append({
                    'epoch': epoch,
                    'loss': loss
                })
                print(f'epoch: {epoch}, loss: {self.loss()}')
        return loss_arr

    def predict(self, inputs):
        hidden_output = self.hidden_layer.forward(inputs)
        return self.output_layer.forward(hidden_output)

    def evaluate(self, inputs, outputs):
        predict_outputs = self.predict(inputs)
        predict_outputs = np.argmax(predict_outputs, axis=1)
        outputs = np.argmax(outputs, axis=1)
        return np.sum(predict_outputs == outputs) / len(outputs)
