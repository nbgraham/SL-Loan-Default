from analysis import test_f_stars, rel
from data import load_loan, split
from random_forest import RandomForestClassifier
from random_forest_sampling_experiment import sample

import numpy as np


def main():
    data, target, attributes = load_loan()
    training_val_data, training_val_target, test_data, test_target = split(data, target)

    clf = RandomForestClassifier(max_depth=5, n_trees=70, show_progress=True, sample=sample)
    clf.fit(training_val_data, training_val_target, attributes)

    # pred = clf.predict_prob_vote(test_data) w/ 100 trees AUC = 0.73
    pred = clf.predict_prob(test_data) # w/100 trees AUC = 0.78

    rel(pred[:, 1], test_target, plot=True)
    f_stars = [i / 100 for i in range(100)]
    auc, max_acc, pod, pofd = test_f_stars(pred, test_target, f_stars)
    print("AUC: ", auc, " Max Accuracy: ", max_acc, " Min Square ROC dist: POD: ", pod, " POFD: ", pofd)

    # change = (pred[:,1]*0.05).reshape(-1, 1)
    # change_a = np.hstack([-1*change, change])
    # pred += change_a
    #
    # rel(pred[:,1], test_target, plot=True)
    # f_stars = [i/100 for i in range(100)]
    # auc, acc = test_f_stars(pred, test_target, f_stars)
    # print("AUC: ", auc, " Accuracy: ", acc)


if __name__ == "__main__":
    main()