from matplotlib import pyplot as plt
import numpy as np


def plot():
    with open('knn_auc_sklearn.npy', 'rb') as f:
        a = np.load(f)

    with open('knn_acc_sklearn.npy', 'rb') as f:
        b = np.load(f)

    _k = [1, 2, 3, 4, 5, 10, 25, 50, 100, 500]
    _weights = [1]

    max_auc = a.max()
    max_run_auc = np.unravel_index(a.argmax(), a.shape)
    print("Max AUC: {} with k: {} and weights: {}".format(max_auc, _k[max_run_auc[0]], _k[max_run_auc[1]]))
    max_acc = b.max()
    max_run_acc = np.unravel_index(b.argmax(),b.shape)
    print("Max Accuracy: {} with k: {} and weights: {}".format(max_acc,_k[max_run_acc[0]],_k[max_run_acc[1]]))

    plt.figure(1)
    plt.plot(range(len(_k)),a,label="AUC")
    plt.plot(range(len(_k)),b,label="Accuracy")
    plt.title("Knn: Accuracy/AUC vs K")
    plt.ylim(0,1)
    plt.ylabel("AUC/Accuracy")
    plt.xlabel("K values (Number of Neighbors)")
    plt.xticks(range(len(_k)), _k)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    plot()
