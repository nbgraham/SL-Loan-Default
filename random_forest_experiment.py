import numpy as np

from random_forest import RandomForestClassifier as MyRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier as SKLearnRandomForestClassifier
from data import load_loan, load_loan_no_history
from model_experiment import _test_model
from wrapper import Wrapper

MY_CODE = True
HISTORY = False


def main():
    data, target, attributes = load_loan() if HISTORY else load_loan_no_history()
    test_random_forest(data, target, attributes)


def test_random_forest(data, target, attributes):
    _n_estimators = [j for j in range(40,100,20)]
    _max_depths = [i for i in range(3,10,2)]
    fs = ['gini', 'entropy']

    auc_total, rel_total = run_experiments(data, target, attributes, 4, fs, _max_depths, _n_estimators)

    save(auc_total, rel_total)


def run_experiments(data, target, attributes, n, *params):
    auc_total = 0
    rel_total = 0

    for i in range(n):
        print("   ---- SUPER RUN {}/{} ----".format(i+1,n))
        experiment_data = _test_model(data, target, attributes, create_random_forest, save, *params)

        auc_total += experiment_data['auc_grid']
        rel_total += experiment_data['rel_grid']

    auc_total /= n
    rel_total /= n

    return auc_total, rel_total

def save(auc_grid, acc_grid):
    suffix = "" if HISTORY else "_no_history"
    with open('random_forest_auc' + suffix + '.npy', 'wb') as auc:
        np.save(auc, auc_grid)

    with open('random_forest_acc' + suffix + '.npy', 'wb') as acc:
        np.save(acc, acc_grid)


def create_random_forest(f, max_depth, n_est):
    print("Creating random forest with max_depth={}; remainder scoring with {}; n_estimators={}".format(max_depth, f, n_est))
    return Wrapper(my_code=MY_CODE, my_model=MyRandomForestClassifier(max_depth=max_depth, remainder_score=f, n_trees=n_est), sklearn_model=SKLearnRandomForestClassifier(max_depth=max_depth, criterion=f, n_estimators=n_est))


if __name__ == "__main__":
    main()