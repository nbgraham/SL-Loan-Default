from analysis import test_f_stars, rel
from data import load_loan, split
from random_forest import RandomForestClassifier

import numpy as np


def main():
    data, target, attributes = load_loan()
    training_val_data, training_val_target, test_data, test_target = split(data, target)

    clf = RandomForestClassifier(max_depth=6, n_trees=1, min_split_size=100, show_progress=True)
    clf.fit(training_val_data, training_val_target, attributes)

    # pred = clf.predict_prob_vote(test_data) w/ 100 trees AUC = 0.73
    pred = clf.predict_prob(test_data) # w/100 trees AUC = 0.78

    rel(pred[:, 1], test_target, plot=True)
    f_stars = [i / 100 for i in range(100)]
    auc, acc = test_f_stars(pred, test_target, f_stars)
    print("AUC: ", auc, " Accuracy: ", acc)

    change = (pred[:,1]*0.05).reshape(-1, 1)
    change_a = np.hstack([-1*change, change])
    pred += change_a

    rel(pred[:,1], test_target, plot=True)
    f_stars = [i/100 for i in range(100)]
    auc, acc = test_f_stars(pred, test_target, f_stars)
    print("AUC: ", auc, " Accuracy: ", acc)


if __name__ == "__main__":
    main()