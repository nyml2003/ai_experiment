import config
from train import NeuralNetwork
import matplotlib.pyplot as plt
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
    loss_arr=nn.train(config.epochs)
    print(loss_arr)
    plt.plot([i['epoch'] for i in loss_arr], [i['loss'] for i in loss_arr])
    plt.show()

    res = nn.evaluate(config.test_x, config.test_y)
    print(f'accuracy: {res}')
