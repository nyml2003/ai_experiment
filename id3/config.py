import os
from util import read_data
data_path = 'data'
train_data_filename = 'traindata.txt'
test_data_filename = 'testdata.txt'
train_data = read_data(os.path.join(data_path, train_data_filename))
test_data = read_data(os.path.join(data_path, test_data_filename))
