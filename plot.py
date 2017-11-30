from matplotlib import pyplot as plt
import numpy as np


def plot_dec_tree():
    plot("Decision tree", "dec_tree")


def plot_random_forest():
    plot("Random forest", "random_forest")
    

def plot(name, file_prefix):
    with open(file_prefix + '_auc.npy', 'rb') as f:
        a = np.load(f)

    gini = a [0]
    entropy = a [1]

    scoring = ['gini', 'entropy']
    min_split_sizes = [j for j in range(50, 150, 20)]
    max_depths = [i for i in range(5, 11)]

    max_auc = a.max()
    max_run = np.unravel_index(a.argmax(), a.shape)
    print("Max AUC: {} with max depth: {} min split size: {} and scoring: {}".format(max_auc, max_depths[max_run[1]], min_split_sizes[max_run[2]], scoring[max_run[0]]))

    plt.imshow(gini)
    plt.colorbar()
    plt.title(name + " with Gini: AUC")
    plt.ylabel("Max depth")
    plt.yticks(range(len(max_depths)), max_depths)
    plt.xlabel("Min split size")
    plt.xticks(range(len(min_split_sizes)), min_split_sizes)
    plt.show()

    plt.imshow(entropy)
    plt.colorbar()
    plt.title(name + " with Entropy: AUC")
    plt.ylabel("Max depth")
    plt.yticks(range(len(max_depths)), max_depths)
    plt.xlabel("Min split size")
    plt.xticks(range(len(min_split_sizes)), min_split_sizes)
    plt.show()


if __name__ == "__main__":
    plot()
