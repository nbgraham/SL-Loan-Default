import math
import numpy as np

from decision_tree import DecisionTreeClassifier
from data import load_loan, load_loan_no_history
from model_experiment import test_model


def main(history=True):
    data, target, attributes = load_loan() if history else load_loan_no_history()

    test_dec_tree(data, target, attributes)


def test_dec_tree(data, target, attributes):
    lower_split_bound = math.ceil(len(data) / 1000)
    _min_samples = [j for j in range(lower_split_bound, 2 * lower_split_bound, 5)]
    _max_depths = [i for i in range(3, len(attributes) * 2)]
    fs = ['gini', 'entropy']

    experiment_data = test_model(data, target, attributes, create_decision_tree, fs, _max_depths, _min_samples)

    with open('dec_tree_auc.npy', 'wb') as auc:
        np.save(auc, experiment_data['auc_grid'])

    with open('dec_tree_acc.npy', 'wb') as acc:
        np.save(acc, experiment_data['acc_grid'])


def create_decision_tree(f, max_depth, min_sm):
    print("Creating decision tree with max_depth={}; remainder scoring with {}; min_split_size={}".format(max_depth, f, min_sm))
    return DecisionTreeClassifier(max_depth=max_depth, remainder_score=f, min_split_size=min_sm)


if __name__ == "__main__":
    main()