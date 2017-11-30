import numpy as np

from random_forest import RandomForestClassifier
from data import load_loan, load_loan_no_history, custom_sample
from model_experiment import _test_model


def sample(x, y, i, n):
    return custom_sample(x, y, 1.1-(i/n), 1+(i/n))


def main(history=True):
    data, target, attributes = load_loan() if history else load_loan_no_history()
    test_random_forest(data, target, attributes)


def test_random_forest(data, target, attributes):
    _n_estimators = [j for j in range(40,100,40)]
    _max_depths = [i for i in range(4,12,4)]
    fs = ['gini', 'entropy']

    experiment_data = _test_model(data, target, attributes, create_random_forest, save, fs, _max_depths, _n_estimators)

    save(experiment_data['auc_grid'], experiment_data['acc_grid'])


def save(auc_grid, acc_grid):
    with open('random_forest_sample_auc.npy', 'wb') as auc:
        np.save(auc, auc_grid)

    with open('random_forest_sample_acc.npy', 'wb') as acc:
        np.save(acc, acc_grid)


def create_random_forest(f, max_depth, n_est):
    print("Creating random forest with max_depth={}; remainder scoring with {}; n_estimators={}".format(max_depth, f, n_est))
    return RandomForestClassifier(max_depth=max_depth, remainder_score=f, n_trees=n_est, sample=sample)


if __name__ == "__main__":
    main()