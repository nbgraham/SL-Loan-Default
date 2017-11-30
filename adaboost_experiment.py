import numpy as np

from adaboost import Adaboost
from data import load_loan, load_loan_no_history
from model_experiment import test_model


def main(history=True):
    data, target, attributes = load_loan() if history else load_loan_no_history()
    test_adaboost(data, target, attributes)


def test_adaboost(data, target, attributes):
    _n_estimators = [j for j in range(10,100,10)]
    _max_depths = [i for i in range(3, len(attributes) * 2)]
    learning_rates = [(i+1)/10 for i in range(10)]

    experiment_data = test_model(data, target, attributes, create_adaboost, _n_estimators, _max_depths, learning_rates)

    with open('adaboost_auc.npy', 'wb') as auc:
        np.save(auc, experiment_data['auc_grid'])

    with open('adaboost_acc.npy', 'wb') as acc:
        np.save(acc, experiment_data['acc_grid'])


def create_adaboost(n_est, max_depth, learning_rate):
    print("Creating adaboost with max_depth={}; learning rate={}; n_estimators={}".format(max_depth, learning_rate, n_est))
    return Adaboost(max_depth=max_depth, learning_rate=learning_rate, n_models=n_est)


if __name__ == "__main__":
    main()