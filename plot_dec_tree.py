from matplotlib import pyplot as plt
import numpy as np

def plot(metric):
    with open(metric + '.npy', 'rb') as f:
        a = np.load(f)

    gini = a [0]
    entropy = a [1]

    min_split_sizes = [j for j in range(30, 100, 8)]
    max_depths = [i for i in range(3, 20)]

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
    plot('acc')