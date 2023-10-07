from util import read_data
data_path = '../data/experiment1/'
train_data_filename = 'traindata.txt'
test_data_filename = 'testdata.txt'
train_data = read_data(data_path + train_data_filename)
test_data = read_data(data_path + test_data_filename)