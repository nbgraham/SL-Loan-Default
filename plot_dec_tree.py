from matplotlib import pyplot as plt
import numpy as np

def plot(metric):
    with open(metric + '.npy', 'rb') as f:
        a = np.load(f)

    gini = a [0]
    entropy = a [1]

    plt.imshow(gini)
    plt.colorbar()
    plt.title("Gini " + metric)
    plt.ylabel("Max depth")
    plt.yticks(range(43), range(3,46))
    plt.xlabel("Min split size")
    plt.xticks(range(6), range(30, 60, 5))
    plt.show()

    plt.imshow(entropy)
    plt.colorbar()
    plt.title("Entropy " + metric)
    plt.ylabel("Max depth")
    plt.xlabel("Min split size")
    plt.show()


if __name__ == "__main__":
    plot('auc')
    # plot('acc')