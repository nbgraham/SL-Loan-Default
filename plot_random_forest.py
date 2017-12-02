from matplotlib import pyplot as plt
import numpy as np


def plot():
    with open('random_forest_auc_no_history.npy', 'rb') as f:
        a = np.load(f)

    gini = a [0]
    entropy = a [1]

    _n_estimators = [j for j in range(40, 100, 20)]
    _max_depths = [i for i in range(4, 12, 2)]
    fs = ['gini', 'entropy']

    max_auc = a.max()
    max_run = np.unravel_index(a.argmax(), a.shape)
    print("Max AUC: {} with max depth: {} min split size: {} and scoring: {}".format(max_auc, _max_depths[max_run[1]], _n_estimators[max_run[2]], fs[max_run[0]]))

    plt.imshow(gini)
    plt.colorbar()
    plt.title("Random Forest Gini: AUC")
    plt.ylabel("Max depth")
    plt.yticks(range(len(_max_depths)), _max_depths)
    plt.xlabel("Number of Trees")
    plt.xticks(range(len(_n_estimators)), _n_estimators)
    plt.show()

    plt.imshow(entropy)
    plt.colorbar()
    plt.title("Random Forest Entropy: AUC")
    plt.ylabel("Max depth")
    plt.yticks(range(len(_max_depths)), _max_depths)
    plt.xlabel("Number of Trees")
    plt.xticks(range(len(_n_estimators)), _n_estimators)
    plt.show()


if __name__ == "__main__":
    plot()
