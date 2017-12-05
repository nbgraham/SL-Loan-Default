import numpy as np

from adaboost import Adaboost
from data import load_loan, load_loan_no_history
from model_experiment import _test_model
from wrapper import Wrapper
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

MY_CODE = False

def main(history=True):
    data, target, attributes = load_loan() if history else load_loan_no_history()
    test_adaboost(data, target, attributes)


def test_adaboost(data, target, attributes):
    _n_estimators = [1,5,10,25,50]
    _max_depths = [1,3,5,10,23]
    _learning_rates = [0.1,0.25,0.5,0.75,1.0]

    experiment_data = _test_model(data, target, attributes, create_adaboost, save, _n_estimators, _max_depths, _learning_rates)

    save(experiment_data['auc_grid'], experiment_data['acc_grid'])


def save(auc_grid, acc_grid):
    if MY_CODE:
        with open('adaboost_auc.npy', 'wb') as auc:
            np.save(auc, auc_grid)

        with open('adaboost_acc.npy', 'wb') as acc:
            np.save(acc, acc_grid)
    else:
        with open('adaboost_comparison_auc.npy', 'wb') as auc:
            np.save(auc, auc_grid)

        with open('adaboost_comparison_acc.npy', 'wb') as acc:
            np.save(acc, acc_grid)


def create_adaboost(n_est, max_depth, learning_rate):
    print("Creating adaboost with max_depth={}; learning rate={}; n_estimators={}".format(max_depth, learning_rate, n_est))
    return Wrapper(my_code=MY_CODE,my_model=Adaboost(max_depth=max_depth, learning_rate=learning_rate, n_models=n_est), sklearn_model=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',max_depth=max_depth),n_estimators=n_est,learning_rate=learning_rate))



if __name__ == "__main__":
    main()