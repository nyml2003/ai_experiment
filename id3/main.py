from train import make_tree
from config import train_data, test_data
from util import calc_accuracy, post_pruning, dict_to_mermaid
from sklearn.model_selection import KFold

if __name__ == '__main__':
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)
    train_data = train_data.sample(frac=1)
    average = 0
    for i, (train_index, test_index) in enumerate(kf.split(train_data)):
        tree = post_pruning(make_tree(train_data.iloc[train_index, :], 0))
        accuracy = calc_accuracy(tree, train_data.iloc[test_index, :])
        average += accuracy
        print(f'Fold{i + 1}:', accuracy)
    print('average:', average / n_splits)
    tree = post_pruning(make_tree(train_data, 0))
    print(tree)
    print('test', calc_accuracy(tree, test_data))
    print(dict_to_mermaid(tree))

