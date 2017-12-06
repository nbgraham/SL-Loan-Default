import numpy as np
from matplotlib import pyplot as plt


def main():
    with open('rel_basic.npy','rb') as f:
        basic = np.load(f)
    with open('rel_slight.npy','rb') as f:
        slight = np.load(f)
    with open('rel_heavy.npy','rb') as f:
        heavy = np.load(f)

    predictions = [(i+.5)/10 for i in range(10)]

    plt.title("Reliability across samplings")
    plt.xlabel("Predicted Event Freq")
    plt.ylabel("Conditional Event Freq")
    plt.plot(predictions, predictions, "--", label="Perfect")
    plt.plot(predictions, basic, label="Original")
    plt.plot(predictions, slight, label="Slight")
    plt.plot(predictions, heavy, label="Substaintial")
    plt.legend()
    
    plt.show()


if __name__ == "__main__":
    main()
