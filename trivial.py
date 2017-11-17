import numpy as np

from analysis import test_f_stars
from data import load_loan, split

f_stars = [i/10 for i in range(10)]


def main():
    data, target, attributes = load_loan()

    training_data, training_target, test_data, test_target = split(data, target)

    pred = np.sum(training_target)/len(training_target)

    best_preds = [[0,1] if pred > 0.5 else [1,0] for t in test_target]
    auc, acc = test_f_stars(best_preds, test_target, f_stars)

    print("Always predict: ", "No" if pred < 0.5 else "Yes")
    print("  AUC: ", auc, " Accuracy: ", acc)

    rand_preds = [[0,1] if np.random.rand() < pred else [1,0] for t in test_target]
    auc, acc = test_f_stars(rand_preds, test_target, f_stars)

    print("Predict yes with prob: ", pred)
    print("  AUC: ", auc, " Accuracy: ", acc)



if __name__ == "__main__":
    main()

