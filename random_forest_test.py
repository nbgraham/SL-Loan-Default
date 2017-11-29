from analysis import test_f_stars
from data import load_loan, split
from random_forest import RandomForestClassifier

import numpy as np
from matplotlib import pyplot as plt

def main():
    data, target, attributes = load_loan()
    training_val_data, training_val_target, test_data, test_target = split(data, target)

    clf = RandomForestClassifier(max_depth=6, n_trees=10, min_split_size=100, show_progress=True)
    clf.fit(training_val_data, training_val_target, attributes)

    # pred = clf.predict_prob_vote(test_data) w/ 100 trees AUC = 0.73
    pred = clf.predict_prob(test_data) # w/100 trees AUC = 0.78

    rel(pred[:,1]*1.05, test_target)

    f_stars = [i/100 for i in range(100)]
    auc, acc = test_f_stars(pred, test_target, f_stars)

    print("AUC: ", auc, " Accuracy: ", acc)


def rel(pred, target):
    cond_event_freqs = []
    predicted_freqs = []

    buckets = 10
    for i in range(buckets):
        prediction_start = i/10
        prediction_end = (i+1)/10

        target_indices_in_prediction_range = (pred >= prediction_start)*(pred < prediction_end)

        cond_event_freq = np.sum(target[target_indices_in_prediction_range])/np.sum(target_indices_in_prediction_range)

        cond_event_freqs.append(cond_event_freq)
        predicted_freqs.append((i+.5)/10)

    perfect = [i/buckets for i in range(buckets)]

    rel = 0
    for i in range(len(cond_event_freqs)):
        rel += (cond_event_freqs[i]-predicted_freqs[i])**2

    print("Reliability = ", rel)

    plt.plot(perfect, perfect, "--", label="Perfect")
    plt.plot(predicted_freqs, cond_event_freqs, label="Model")
    plt.title("Reliability")
    plt.xlabel("Predicted Event Freq")
    plt.ylabel("Conditional Event Freq")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()