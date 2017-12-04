from analysis import test_f_stars, rel, inv_rel
from data import load_loan, split
from random_forest import RandomForestClassifier
from random_forest_sampling_experiment import sample

import numpy as np


def main():
    data, target, attributes = load_loan()
    training_val_data, training_val_target, test_data, test_target = split(data, target)

    clf = RandomForestClassifier(max_depth=5, n_trees=20, show_progress=True)
    clf.fit(training_val_data, training_val_target, attributes)

    # pred = clf.predict_prob_vote(test_data) w/ 100 trees AUC = 0.73
    pred = clf.predict_prob(test_data) # w/100 trees AUC = 0.78

    inv_rel(pred[:, 1], test_target)
    rel(pred[:, 1], test_target, plot=True)

    f_stars = [i / 100 for i in range(100)]
    auc, max_acc, pod, pofd = test_f_stars(pred, test_target, f_stars)
    print("\nAUC: ", auc, " Max Accuracy: ", max_acc, " Min Square ROC dist: POD: ", pod, " POFD: ", pofd)

    # change = (pred[:,1]*0.05).reshape(-1, 1)
    # change_a = np.hstack([-1*change, change])
    # pred += change_a
    #
    # rel(pred[:,1], test_target, plot=True)
    # f_stars = [i/100 for i in range(100)]
    # auc, acc = test_f_stars(pred, test_target, f_stars)
    # print("AUC: ", auc, " Accuracy: ", acc)


def best_rel():
    data, target, attributes = load_loan()
    training_val_data, training_val_target, test_data, test_target = split(data, target)
    training_data, training_target, val_data, val_target = split(training_val_data, training_val_target)

    clf = RandomForestClassifier(max_depth=5, n_trees=70, show_progress=True)
    clf.fit(training_data, training_target, attributes)

    test_pred = clf.predict_prob(test_data)
    orig_test_rel = rel(test_pred[:, 1], test_target)

    val_pred = clf.predict_prob(val_data) # w/100 trees AUC = 0.78
    min_rel = 1
    best_b = 0
    best_m = 1
    for b in range(0,10):
        b = 0.05 - b/100
        for m in range(0, 11):
            m= 1 + (m-5)/30
            cur_rel = rel(m*val_pred[:, 1]+b, val_target)
            print("m: {} b: {} rel: {}".format(m,b,cur_rel))
            if cur_rel < min_rel:
                min_rel = cur_rel
                best_b = b
                best_m = m
    print("\nBest m: {} b: {} with rel: {}".format(best_m, best_b, min_rel))

    test_pred = clf.predict_prob(test_data)
    test_pred = best_m * test_pred + best_b
    test_rel = rel(test_pred[:,1], test_target)
    print("Original Test Rel: ", orig_test_rel, "Test Rel: ", test_rel)

    f_stars = [i/100 for i in range(100)]
    auc, acc, pod, pofd = test_f_stars(test_pred, test_target, f_stars)
    print("AUC: ", auc, " Accuracy: ", acc," POD: ", pod, " POFD: ", pofd)


if __name__ == "__main__":
    main()