import json
import numpy as np
from matplotlib import pyplot as plt


def test_f_stars(pred, true, f_stars, status_delay=100, filename='model'):
    pods = []
    pofds = []

    max_acc = 0
    max_f = None
    for f_star in f_stars:
        if (f_star * 1000) % status_delay == 0:
            print(f_star)
        acc, pod, pofd = analyze(pred, true, f_star)
        if acc > max_acc:
            max_acc = acc
            max_f = f_star

        pods.append(pod)
        pofds.append(pofd)

    with open(filename + '_pods.json','w') as f:
        json.dump(pods, f)

    with open(filename + '_pofds.json','w') as f:
        json.dump(pofds, f)

    auc = -1 * np.trapz(y=pods, x=pofds)

    print("AUC: {}".format(auc))

    print("Max acc")
    print(max_acc)
    print(max_f)

    plot_roc(filename, pofds, pods)


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
        pred_prob = pred_probs[i]

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