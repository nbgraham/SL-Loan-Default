import numpy as np

from knn import KnnClassifier
from data import load_loan, load_loan_no_history
from model_experiment import _test_model


def main(history=True):
    data, target, attributes = load_loan() if history else load_loan_no_history()
    test_knn(data, target, attributes)


def test_knn(data, target, attributes):
    _k = [1,2,3,4,5,10,25,50,100,250,500,1000] # picked some arbitrary k values
    # _weights = [[1] for _ in attributes] # idk what I'm doing here
    _weights = [1]

    experiment_data = _test_model(data, target, attributes, create_knn, save, _k, _weights)

    save(experiment_data['auc_grid'], experiment_data['acc_grid'])


def save(auc_grid, acc_grid):
    with open('knn_auc.npy', 'wb') as auc:
        np.save(auc, auc_grid)

    with open('knn_acc.npy', 'wb') as acc:
        np.save(acc, acc_grid)


def create_knn(k, weights):
    print("Creating knn classifer with k={}; weights={}".format(k, weights))
    return KnnClassifier(k=k, weights=weights)


if __name__ == "__main__":
    main()