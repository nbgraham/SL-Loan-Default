import numpy as np
import threading
import math

from data import load_loan
from analysis import test_f_stars
from data import split
from decision_tree import DecisionTreeClassifier

n_fs = 100
f_stars = [i/n_fs for i in range(n_fs)]


def main():
    data, target, attributes = load_loan()

    lower_split_bound = math.ceil(len(data) / 1000)
    min_split_sizes = [j for j in range(lower_split_bound, 2 * lower_split_bound, 5)]
    max_depths = [i for i in range(3, len(attributes) * 2)]
    remainder_scores = ['gini', 'entropy']

    auc_grid, acc_grid = test_params(data, target, attributes, remainder_scores, max_depths, min_split_sizes)

    with open('auc.npy', 'wb') as auc:
        np.save(auc, auc_grid)

    with open('acc.npy', 'wb') as acc:
        np.save(acc, acc_grid)


def test_params(data, target, attributes, remainder_scores, max_depths, min_split_sizes):
    max_max_depths = np.max(max_depths)
    min_min_sizes = np.min(min_split_sizes)

    auc_grid = np.empty((len(remainder_scores), len(max_depths), len(min_split_sizes)))
    acc_grid = np.empty((len(remainder_scores), len(max_depths), len(min_split_sizes)))

    training_val_data, training_val_target, test_data, test_target = split(data, target)
    training_data, training_target, val_data, val_target = split(training_val_data, training_val_target)

    threads = []
    progress = {
        'total': len(remainder_scores) * len(max_depths) * len(min_split_sizes),
        'i': 0
    }

    ## Training data is the same for all trees!!! Loses some exploration/randomness
    for remainder_score_i in range(len(remainder_scores)):
        remainder_score = remainder_scores[remainder_score_i]

        t = threading.Thread(target=test_rem_score, args=(progress, remainder_score, max_max_depths, min_min_sizes, training_data, training_target, val_data, val_target, attributes, remainder_score_i, max_depths, min_split_sizes, auc_grid, acc_grid))
        threads.append(t)
        t.start()

    for thread in threads:
        thread.join()

    return auc_grid, acc_grid


def test_rem_score(progress, remainder_score, max_max_depths, min_min_sizes, training_data, training_target, val_data, val_target, attributes, remainder_score_i, max_depths, min_split_sizes, auc_grid, acc_grid):
    clf = DecisionTreeClassifier(remainder_score=remainder_score, max_depth=max_max_depths,
                                 min_split_size=min_min_sizes)
    clf.fit(training_data, training_target, attributes)

    threads = []
    for max_depth_i in range(len(max_depths)):
        for min_split_size_i in range(len(min_split_sizes)):
            max_depth = max_depths[max_depth_i]
            min_split_size = min_split_sizes[min_split_size_i]

            progress['i'] += 1
            print("Run {}/{}".format(progress['i'], progress['total']))

            if len(threads) >= 16:
                threads[0].join()
                threads = threads[1:]
            t = threading.Thread(target=predict_one, args=(
            clf, val_data, val_target, max_depth, min_split_size, auc_grid, acc_grid, remainder_score_i, max_depth_i,
            min_split_size_i))
            threads.append(t)
            t.start()

    for thread in threads:
        thread.join()


def predict_one(clf, val_data, val_target, max_depth, min_split_size, auc_grid, acc_grid, remainder_score_i, max_depth_i, min_split_size_i):
    val_preds = clf._predict_prob(val_data, max_depth, min_split_size)
    auc, max_acc = test_f_stars(val_preds, val_target, f_stars, status_delay=25)

    auc_grid[(remainder_score_i, max_depth_i, min_split_size_i)] = auc
    acc_grid[(remainder_score_i, max_depth_i, min_split_size_i)] = max_acc


if __name__ == "__main__":
    main()
