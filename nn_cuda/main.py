import config
from train import NeuralNetwork

if __name__ == '__main__':
    nn = NeuralNetwork(
        config.train_x,
        config.train_y,
        config.input_size,
        config.hidden_size,
        config.output_size,
        config.seed,
        config.learning_rate
    )
    nn.train(config.epochs)
    res = nn.evaluate(config.test_x, config.test_y)
    print('Accuracy: {:.2f}%'.format(res * 100))
