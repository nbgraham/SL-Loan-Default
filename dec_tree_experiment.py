import math

from decision_tree import DecisionTreeClassifier
from data import load_loan
from model_experiment import test_model


def main():
    data, target, attributes = load_loan()

    test_dec_tree(data, target, attributes)


def test_dec_tree(data, target, attributes):
    lower_split_bound = math.ceil(len(data) / 1000)
    _min_samples = [j for j in range(lower_split_bound, 2 * lower_split_bound, 5)]
    _max_depths = [i for i in range(3, len(attributes) * 2)]
    fs = ['gini', 'entropy']

    test_model(data, target, attributes, create_decision_tree, _max_depths, fs, _min_samples)


def create_decision_tree(max_depth, f, min_sm):
    print("Creating decision tree with max_depth={}; remainder scoring with {}; min_split_size={}".format(max_depth, f, min_sm))
    return DecisionTreeClassifier(max_depth=max_depth, remainder_score=f, min_split_size=min_sm)


if __name__ == "__main__":
    main()