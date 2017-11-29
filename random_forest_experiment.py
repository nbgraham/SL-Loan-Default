import math
import numpy as np

from random_forest import RandomForestClassifier
from data import load_loan
from model_experiment import test_model


def main():
    data, target, attributes = load_loan()

    test_dec_tree(data, target, attributes)


def test_dec_tree(data, target, attributes):
    _n_estimators = [j for j in range(10,100,10)]
    _max_depths = [i for i in range(3, len(attributes) * 2)]
    fs = ['gini', 'entropy']

    experiment_data = test_model(data, target, attributes, create_random_forest, fs, _max_depths, _n_estimators)

    with open('auc.npy', 'wb') as auc:
        np.save(auc, experiment_data['auc_grid'])

    with open('acc.npy', 'wb') as acc:
        np.save(acc, experiment_data['acc_grid'])


def create_random_forest(f, max_depth, n_est):
    print("Creating random forest with max_depth={}; remainder scoring with {}; n_estimators={}".format(max_depth, f, n_est))
    return RandomForestClassifier(max_depth=max_depth, remainder_score=f, n_trees=n_est)


if __name__ == "__main__":
    main()