import numpy as np

from knn import KnnClassifier
from data import load_loan, load_loan_no_history
from model_experiment import _test_model
from sklearn.utils.extmath import cartesian
from wrapper import Wrapper
from sklearn.neighbors import KNeighborsClassifier

MY_CODE = False

def main(history=True):
    data, target, attributes = load_loan() if history else load_loan_no_history()
    test_knn(data, target, attributes)


def test_knn(data, target, attributes):
    _k = [1,2,3,4,5,10,25,50,100,500]
    _weights = [1] #uniform weights only
    # possible_weights = [j/4-1 for j in range(0,9)]
    # a = tuple(possible_weights for _ in range(len(attributes)))
    # _weights = cartesian(a)

    experiment_data = _test_model(data, target, attributes, create_knn, save, _k, _weights)
    # experiment_data = _test_model(data, target, attributes, create_knn, save, _k, np.random.choice(_weights,10,replace=False))

    save(experiment_data['auc_grid'], experiment_data['acc_grid'])


def save(auc_grid, acc_grid):
    if MY_CODE:
        with open('knn_auc.npy', 'wb') as auc:
            np.save(auc, auc_grid)

        with open('knn_acc.npy', 'wb') as acc:
            np.save(acc, acc_grid)
    else:
        with open('knn_comparison_auc.npy', 'wb') as auc:
            np.save(auc, auc_grid)

        with open('knn_comparison_acc.npy', 'wb') as acc:
            np.save(acc, acc_grid)


def create_knn(k, weights):
    print("Creating knn classifier with k={}; weights={}".format(k, weights))
    return Wrapper(my_code=MY_CODE,my_model=KnnClassifier(k=k, weights=weights),sklearn_model=KNeighborsClassifier(n_neighbors=k,weights='uniform'))


if __name__ == "__main__":
    main()