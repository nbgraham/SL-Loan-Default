from matplotlib import pyplot as plt
import numpy as np

def plot(metric):
    with open(metric + '.npy', 'rb') as f:
        a = np.load(f)

    gini = a [0]
    entropy = a [1]

    scoring = ['gini', 'entropy']
    min_split_sizes = [j for j in range(40, 200, 20)]
    max_depths = [i for i in range(3, 20)]

    max_auc = a.max()
    max_run = np.unravel_index(a.argmax(), a.shape)
    print("Max AUC: {} with max depth: {} min split size: {} and scoring: {}".format(max_auc, max_depths[max_run[1]], min_split_sizes[max_run[2]], scoring[max_run[0]]))

    plt.imshow(gini)
    plt.colorbar()
    plt.title("Gini " + metric)
    plt.ylabel("Max depth")
    plt.yticks(range(len(max_depths)), max_depths)
    plt.xlabel("Min split size")
    plt.xticks(range(len(min_split_sizes)), min_split_sizes)
    plt.show()

    plt.imshow(entropy)
    plt.colorbar()
    plt.title("Entropy " + metric)
    plt.ylabel("Max depth")
    plt.yticks(range(len(max_depths)), max_depths)
    plt.xlabel("Min split size")
    plt.xticks(range(len(min_split_sizes)), min_split_sizes)
    plt.show()


if __name__ == "__main__":
    plot('auc')
    # plot('acc')