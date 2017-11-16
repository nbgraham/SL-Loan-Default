import math

from data import load_loan, split
from decision_tree import DecisionTreeClassifier
from analysis import test_f_stars

n_fs = 100
f_stars = [i/n_fs for i in range(n_fs)]


def main():
    data, target, attributes = load_loan()

    training_val_data, training_val_target, test_data, test_target = split(data, target)

    lower_split_bound = math.ceil(len(data)/1000)
    _min_samples = [j for j in range(lower_split_bound,2*lower_split_bound,5)]
    _max_depths = [i for i in range(3,len(attributes)*2)]
    fs = ['gini', 'entropy']

    max_auc = 0
    total = len(_min_samples) * len(_max_depths) * len(fs)
    i = 0
    for min_sm in _min_samples:
        for f in fs:
            for max_depth in _max_depths:
                i += 1
                print("Run {}/{}".format(i, total))

                training_data, training_target, val_data, val_target = split(training_val_data, training_val_target)

                clf = DecisionTreeClassifier(max_depth=max_depth, remainder_score=f, min_split_size=min_sm)
                clf.fit(training_data, training_target, attributes)

                val_preds = clf.predict_prob(val_data)

                auc = test_f_stars(val_preds, val_target, f_stars, status_delay=25)
                if auc > max_auc:
                    max_auc = auc
                    print("Max AUC: {} with min samples: {} f: {} max depth: {}".format(auc, min_sm, f, max_depth))

if __name__ == "__main__":
    main()