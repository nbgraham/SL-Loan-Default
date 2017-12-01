import numpy as np

from knn import KnnClassifier
from data import load_loan, load_loan_no_history
from model_experiment import _test_model


def main(history=True):
    data, target, attributes = load_loan() if history else load_loan_no_history()
    test_knn(data, target, attributes)


def test_knn(data, target, attributes):
    _k = [k for k in range(1,90,15)]
    # _weights = [i for i in range(3, len(attributes) * 2, 5)]

    # experiment_data = _test_model(data, target, attributes, create_knn, save, _k, _weights)
    experiment_data = _test_model(data, target, attributes, create_knn, save, _k)

    save(experiment_data['auc_grid'], experiment_data['acc_grid'])


def save(auc_grid, acc_grid):
    with open('knn_auc.npy', 'wb') as auc:
        np.save(auc, auc_grid)

    with open('knn_acc.npy', 'wb') as acc:
        np.save(acc, acc_grid)


def create_knn(k, weights):
    print("Creating knn classifer with k={}".format(k))
    return KnnClassifier(k=k)
    # print("Creating knn classifer with k={}; weights={}".format(k, weights))
    # return KnnClassifier(k=k, weights=weights)


if __name__ == "__main__":
    main()