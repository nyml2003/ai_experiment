import os
from util import read_data, normalize, one_hot_encode, sigmoid_derivative, sigmoid

data_path = 'data'
train_data_filename = 'Iris-train.txt'
train_x = read_data(os.path.join(data_path, train_data_filename))
train_y = train_x.pop('class')
train_x = normalize(train_x, method='z_score').values
train_y = one_hot_encode(train_y).values
test_data_filename = 'Iris-test.txt'
test_x = read_data(os.path.join(data_path, test_data_filename))
test_y = test_x.pop('class')
test_x = normalize(test_x, method='z_score').values
test_y = one_hot_encode(test_y).values
learning_rate = 0.01  # 学习率
epochs = 1000  # 最大迭代次数
input_size = 4
hidden_size = 10
output_size = 3
seed = 114
activation = sigmoid
activation_derivative = sigmoid_derivative
