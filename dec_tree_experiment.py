import math
import numpy as np

from decision_tree import DecisionTreeClassifier as MyDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier as SKLearnDecisionTreeClassifier
from data import load_loan, load_loan_no_history
from model_experiment import _test_model
from wrapper import Wrapper

MY_CODE = True


def main(history=True):
    data, target, attributes = load_loan() if history else load_loan_no_history()

    test_dec_tree(data, target, attributes)


def test_dec_tree(data, target, attributes):
    lower_split_bound = math.ceil(len(data) / 1000)
    _min_samples = [j for j in range(50, 150, 20)]
    _max_depths = [i for i in range(5,11)]
    fs = ['gini', 'entropy']

    experiment_data = _test_model(data, target, attributes, create_decision_tree, save, fs, _max_depths, _min_samples)

    save(experiment_data['auc_grid'], experiment_data['acc_grid'])


def save(auc_grid, acc_grid):
    with open('dec_tree_auc.npy', 'wb') as auc:
        np.save(auc, auc_grid)

    with open('dec_tree_acc.npy', 'wb') as acc:
        np.save(acc, acc_grid)


def create_decision_tree(f, max_depth, min_sm):
    print("Creating decision tree with max_depth={}; remainder scoring with {}; min_split_size={}".format(max_depth, f, min_sm))
    return Wrapper(my_code=MY_CODE, my_model=MyDecisionTreeClassifier(max_depth=max_depth, remainder_score=f, min_split_size=min_sm), sklearn_model=SKLearnDecisionTreeClassifier(criterion=f, max_depth=max_depth, min_samples_split=min_sm))


if __name__ == "__main__":
    main()