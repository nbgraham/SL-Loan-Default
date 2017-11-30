import json
import numpy as np
from matplotlib import pyplot as plt


def test_f_stars(pred, true, f_stars, status_delay=100, verbose=False, filename='model'):
    pods = []
    pofds = []

    max_acc = 0
    max_f = None
    for f_star in f_stars:
        if verbose and (f_star * len(f_stars)) % status_delay == 0:
            print(f_star, end=', ')
        acc, pod, pofd = analyze(pred, true, f_star)
        if acc > max_acc:
            max_acc = acc
            max_f = f_star

        pods.append(pod)
        pofds.append(pofd)

    # with open(filename + '_pods.json','w') as f:
    #     json.dump(pods, f)
    #
    # with open(filename + '_pofds.json','w') as f:
    #     json.dump(pofds, f)

    auc = -1 * np.trapz(y=pods, x=pofds)

    if verbose:
        print("\nMax acc: {} at f of {}".format(max_acc, max_f))

    # plot_roc(filename, pofds, pods)

    return auc, max_acc


def plot_roc(filename, pofds=None, pods=None):
    if pofds is None:
        with open(filename + '_pofds.json', 'r') as f:
            pofds = json.load(f)

    if pods is None:
        with open(filename + '_pods.json', 'r') as f:
            pods = json.load(f)

    plt.plot(pofds, pods, label='model')
    plt.title('ROC')
    plt.xlabel('POFD')
    plt.ylabel('POD')

    plt.plot([0,1],[0,1],"--",label='random')
    plt.legend()

    plt.show()


def analyze(pred_probs, true, f_star):
    total = len(pred_probs)
    happened_predicted = 0
    happened_not_predicted = 0
    not_happened_predicted = 0
    not_happened_not_predicted = 0

    for i in range(len(pred_probs)):
        obs = true[i]
        pred_prob = pred_probs[i][1]

        pred_label = 1 if pred_prob >= f_star else 0

        if 1 == obs:
            if pred_label == obs:
                happened_predicted += 1
            else:
                happened_not_predicted += 1
        else:
            if pred_label == obs:
                not_happened_not_predicted += 1
            else:
                not_happened_predicted += 1

    right = happened_predicted + not_happened_not_predicted
    acc = right / total
    pod = happened_predicted / (happened_predicted + happened_not_predicted)
    pofd = not_happened_predicted / (not_happened_predicted + not_happened_not_predicted)

    return (acc, pod, pofd)


def rel(pred, target, plot=False):
    cond_event_freqs = []
    predicted_freqs = []

    buckets = 10
    for i in range(buckets):
        prediction_start = i/10
        prediction_end = (i+1)/10

        target_indices_in_prediction_range = (pred >= prediction_start)*(pred < prediction_end)

        avg_prediction = (i+.5)/10
        instances = np.sum(target_indices_in_prediction_range)
        if instances > 0:
            cond_event_freq = np.sum(target[target_indices_in_prediction_range])/instances
        else:
            cond_event_freq = avg_prediction

        cond_event_freqs.append(cond_event_freq)
        predicted_freqs.append(avg_prediction)

    perfect = [i/buckets for i in range(buckets)]

    rel = 0
    for i in range(len(cond_event_freqs)):
        rel += (cond_event_freqs[i]-predicted_freqs[i])**2

    if plot:
        print("Reliability = ", rel)

        plt.plot(perfect, perfect, "--", label="Perfect")
        plt.plot(predicted_freqs, cond_event_freqs, label="Model")
        plt.title("Reliability")
        plt.xlabel("Predicted Event Freq")
        plt.ylabel("Conditional Event Freq")
        plt.legend()
        plt.show()

    return rel