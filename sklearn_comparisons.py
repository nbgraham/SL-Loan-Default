from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from data import load_loan, split
from analysis import rel, test_f_stars


def main():
    data, target, attributes = load_loan()
    training_val_data, training_val_target, test_data, test_target = split(data, target)

    clf = SVC()
    clf.fit(training_val_data, training_val_target)
    pred = clf.predict_proba(test_data) # w/100 trees AUC = 0.78

    rel(pred[:, 1], test_target, plot=True)
    f_stars = [i / 100 for i in range(100)]
    auc, max_acc, pod, pofd = test_f_stars(pred, test_target, f_stars)
    print("AUC: ", auc, " Max Accuracy: ", max_acc, " Min Square ROC dist: POD: ", pod, " POFD: ", pofd)


if __name__ == "__main__":
    main()