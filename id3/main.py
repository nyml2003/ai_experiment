from train import make_tree
from config import train_data, test_data
from util import calc_accuracy
if __name__ == '__main__':
    train_data_cols=len(train_data[0])
    tree = make_tree(train_data, list(range(train_data_cols - 1)),None,0)
    print(calc_accuracy(tree, test_data))