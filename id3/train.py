from typing import List
from collections import Counter
from math import log2


def retain_column_with_value(data: List[List[float]], column: int, value: float):
    return [item for item in data if item[column] == value]


def retain_column_less_value(data: List[List[float]], column: int, value: float):
    return [item for item in data if item[column] < value]


def retain_column_greater_value(data: List[List[float]], column: int, value: float):
    return [item for item in data if item[column] > value]


def calculate_entropy(data: List[List[float]]) -> float:
    data_num = len(data)
    target_count = Counter([item[-1] for item in data])
    return -sum([(target_count[key] / data_num) * log2(target_count[key] / data_num) for key in target_count])


def gain_continuous(data: List[List[float]], column: int):
    data_num = len(data)
    data_entropy = calculate_entropy(data)
    column_sorted = sorted(set([item[column] for item in data]))
    split_points_entropy = [
        (len(retain_column_less_value(data, column, split_point)) / data_num) * calculate_entropy(
            retain_column_less_value(data, column, split_point))
        for split_point in column_sorted[1:]
    ]
    if len(split_points_entropy) == 0:
        return 0.0, None
    min_split_point_entropy, min_split_point_index = min(
            [(split_points_entropy[i], i) for i in range(len(split_points_entropy))])
    return data_entropy - min_split_point_entropy, (
            column_sorted[min_split_point_index] + column_sorted[min_split_point_index - 1]) / 2


def gain_discrete(data: List[List[float]], column: int) -> float:
    data_num = len(data)
    target_count = Counter([item[column] for item in data])
    return calculate_entropy(data) - sum([
        (target_count[key] / data_num) * calculate_entropy(retain_column_with_value(data, column, key))
        for key in target_count
    ])


def make_tree(data: List[List[float]], columns: List[int], last_column, depth):
    if depth > 16:
        return 'out of length'
    if len(data) == 0:
        return None
    if len(Counter([item[-1] for item in data])) == 1:
        return data[0][-1]
    use_columns = columns.copy()
    if last_column in columns:
        use_columns.remove(last_column)
    (max_gain, max_label), max_index = max([(gain_continuous(data, i), i) for i in use_columns])
    max_label = round(max_label, 2)
    root = {f'{max_index}<{max_label}': {}, f'{max_index}>{max_label}': {}}
    if abs(max_gain) < 1e-6:
        return data[0][-1]
    else:
        root[f'{max_index}<{max_label}'] = make_tree(retain_column_less_value(data, max_index, max_label), columns,
                                                     max_index, depth + 1)
        root[f'{max_index}>{max_label}'] = make_tree(retain_column_greater_value(data, max_index, max_label), columns,
                                                     max_index, depth + 1)
    return root
