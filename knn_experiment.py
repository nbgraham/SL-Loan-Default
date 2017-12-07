import numpy as np

from knn import KnnClassifier
from data import load_loan, load_loan_no_history
from model_experiment import _test_model
from wrapper import Wrapper
from sklearn.neighbors import KNeighborsClassifier

MY_CODE = True
VALIDATION_RUNS = 5

def main(history=True):
    data, target, attributes = load_loan() if history else load_loan_no_history()
    test_knn(data, target, attributes)


def test_knn(data, target, attributes):
    _k = [1,2,3,4,5,10,25,50,100,500]
    _weights = [1] #uniform weights only

    auc_total, rel_total = run_experiments(data, target, attributes, VALIDATION_RUNS, _k, _weights)

    save(auc_total, rel_total)


def run_experiments(data, target, attributes, n, *params):
    auc_total = 0
    rel_total = 0

    for i in range(n):
        print("   ---- SUPER RUN {}/{} ----".format(i+1,n))
        experiment_data = _test_model(data, target, attributes, create_knn, save, *params)

        auc_total += experiment_data['auc_grid']
        rel_total += experiment_data['rel_grid']

    auc_total /= n
    rel_total /= n

    return auc_total, rel_total

def save(auc_grid, acc_grid):
    if MY_CODE:
        with open('knn_auc.npy', 'wb') as auc:
            np.save(auc, auc_grid)

        with open('knn_acc.npy', 'wb') as acc:
            np.save(acc, acc_grid)
    else:
        with open('knn_auc_sklearn.npy', 'wb') as auc:
            np.save(auc, auc_grid)

        with open('knn_acc_sklearn.npy', 'wb') as acc:
            np.save(acc, acc_grid)


def create_knn(k, weights):
    print("Creating knn classifier with k={}; weights={}".format(k, weights))
    return Wrapper(my_code=MY_CODE,my_model=KnnClassifier(k=k, weights=weights),sklearn_model=KNeighborsClassifier(n_neighbors=k,weights='uniform'))


if __name__ == "__main__":
    main()