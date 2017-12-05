from matplotlib import pyplot as plt
import numpy as np


def plot():
    with open('adaboost_auc.npy', 'rb') as f:
        a = np.load(f)
    with open('adaboost_acc.npy', 'rb') as f:
        b = np.load(f)

    one = a [0]
    five = a [1]
    ten = a[2]
    twentyfive = a[3]
    fifty = a[4]

    _n_estimators = [1, 5, 10, 25, 50]
    _max_depths = [1, 3, 5, 10, 23]
    _learning_rates = [0.1, 0.25, 0.5, 0.75, 1.0]

    max_auc = a.max()
    max_run_auc = np.unravel_index(a.argmax(), a.shape)
    print("Max AUC: {} with max depth: {} n_estimators: {} and learning rate: {}".format(max_auc, _max_depths[max_run_auc[1]], _n_estimators[max_run_auc[2]], _learning_rates[max_run_auc[0]]))
    max_acc = b.max()
    max_run_acc = np.unravel_index(b.argmax(), b.shape)
    print("Max Accuracy: {} with max depth: {} n_estimators: {} and learning rate: {}".format(max_acc, _max_depths[max_run_acc[1]], _n_estimators[max_run_acc[2]], _learning_rates[max_run_acc[0]]))


    plt.figure(1)
    plt.imshow(one)
    plt.colorbar()
    plt.title("Adaboost Single Tree: AUC")
    plt.ylabel("Max depth")
    plt.yticks(range(len(_max_depths)), _max_depths)
    plt.xlabel("Learning Rate")
    plt.xticks(range(len(_learning_rates)), _learning_rates)
    # plt.show()

    plt.figure(2)
    plt.imshow(five)
    plt.colorbar()
    plt.title("Adaboost 5 Trees: AUC")
    plt.ylabel("Max depth")
    plt.yticks(range(len(_max_depths)), _max_depths)
    plt.xlabel("Learning Rate")
    plt.xticks(range(len(_learning_rates)), _learning_rates)
    # plt.show()

    plt.figure(3)
    plt.imshow(ten)
    plt.colorbar()
    plt.title("Adaboost 10 Trees: AUC")
    plt.ylabel("Max depth")
    plt.yticks(range(len(_max_depths)), _max_depths)
    plt.xlabel("Learning Rate")
    plt.xticks(range(len(_learning_rates)), _learning_rates)
    # plt.show()

    plt.figure(4)
    plt.imshow(twentyfive)
    plt.colorbar()
    plt.title("Adaboost 25 Trees: AUC")
    plt.ylabel("Max depth")
    plt.yticks(range(len(_max_depths)), _max_depths)
    plt.xlabel("Learning Rate")
    plt.xticks(range(len(_learning_rates)), _learning_rates)
    # plt.show()

    plt.figure(5)
    plt.imshow(fifty)
    plt.colorbar()
    plt.title("Adaboost 50 Trees: AUC")
    plt.ylabel("Max depth")
    plt.yticks(range(len(_max_depths)), _max_depths)
    plt.xlabel("Learning Rate")
    plt.xticks(range(len(_learning_rates)), _learning_rates)
    plt.show()

if __name__ == "__main__":
    plot()
