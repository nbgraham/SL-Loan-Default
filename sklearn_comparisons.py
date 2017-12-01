from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np

from data import load_loan, split
from analysis import rel, test_f_stars


def main():
    data, target, attributes = load_loan()
    training_val_data, training_val_target, test_data, test_target = split(data, target)
    pred = predict(training_val_data, training_val_target, test_data, name='svc')

    rel(pred[:,1], test_target, plot=True)

    f_stars = [i / 100 for i in range(100)]
    auc, max_acc, pod, pofd = test_f_stars(pred, test_target, f_stars, plot=True)
    print("AUC: ", auc, " Max Accuracy: ", max_acc, " Min Square ROC dist: POD: ", pod, " POFD: ", pofd)


def predict(training_val_data, training_val_target, test_data, name='svc'):
    if name == 'svc':
        clf = SVC()
        clf.fit(training_val_data, training_val_target)
        prob1 = clf.predict(test_data)

        prob0 = np.ones(prob1.shape) - prob1
        pred = np.hstack([prob0.reshape(-1, 1), prob1.reshape(-1, 1)])
    elif name == 'mlp':
        clf = MLPClassifier()
        clf.fit(training_val_data, training_val_target)
        pred = clf.predict(test_data)

    return pred

if __name__ == "__main__":
    main()