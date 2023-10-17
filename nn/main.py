import numpy as np

import config
from train import NeuralNetwork
import matplotlib.pyplot as plt

if __name__ == '__main__':
    res_arr=[]
    for i in range(config.runs):
        seed = config.seed * (i + 1)
        nn = NeuralNetwork(
            config.train_x,
            config.train_y,
            config.input_size,
            config.hidden_size,
            config.output_size,
            seed,
            config.learning_rate
        )
        loss_arr = nn.train(config.epochs)
        plt.plot([i['epoch'] for i in loss_arr], [i['loss'] for i in loss_arr])
        plt.show()
        res = nn.evaluate(config.test_x, config.test_y)
        print(f'accuracy: {res}')
        res_arr.append(res)
    print(f'average accuracy: {sum(res_arr) / len(res_arr)}')
    print(f'std: {np.std(res_arr)}')
