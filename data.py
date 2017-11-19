import csv
import numpy as np
import math

from decision_tree import Attribute

np.random.seed(1)


def load_loan():
    with open(r'/home/nick/Downloads/default of credit card clients.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        next(reader, None)

        rows = []
        targets = []
        for row in reader:
            # Exclude ID column
            rows.append(np.array(row[1:-1], dtype=int))
            targets.append(row[-1])

    data = np.vstack(rows)
    target = np.array(targets, dtype=int)
    attributes = [
        Attribute("Balance Limit", False),
        Attribute("Sex", True),
        Attribute("Education", True),
        Attribute("Marriage", True),
        Attribute("Age", False),
        Attribute("Payment 1 months late", False),
        Attribute("Payment 2 months late", False),
        Attribute("Payment 3 months late", False),
        Attribute("Payment 4 months late", False),
        Attribute("Payment 5 months late", False),
        Attribute("Payment 6 months late", False),
        Attribute("Bill 1 amount", False),
        Attribute("Bill 2 amount", False),
        Attribute("Bill 3 amount", False),
        Attribute("Bill 4 amount", False),
        Attribute("Bill 5 amount", False),
        Attribute("Bill 6 amount", False),
        Attribute("Payment 1 amount", False),
        Attribute("Payment 2 amount", False),
        Attribute("Payment 3 amount", False),
        Attribute("Payment 4 amount", False),
        Attribute("Payment 5 amount", False),
        Attribute("Payment 6 amount", False),
    ]

    return data, target, attributes


def load_loan_no_history():
    with open(r'/home/nick/Downloads/default of credit card clients.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        next(reader, None)

        rows = []
        targets = []
        for row in reader:
            # Exclude ID column and history
            rows.append(np.array(row[1:6], dtype=int))
            targets.append(row[-1])

    data = np.vstack(rows)
    target = np.array(targets, dtype=int)
    attributes = [
        Attribute("Balance Limit", False),
        Attribute("Sex", True),
        Attribute("Education", True),
        Attribute("Marriage", True),
        Attribute("Age", False),
    ]

    return data, target, attributes


def split(data, target, training=0.8):
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    split = math.ceil(training*len(data))

    training_data = data[indices[:split]]
    training_target = target[indices[:split]]

    test_data = data[indices[split:]]
    test_target = target[indices[split:]]

    return training_data, training_target, test_data, test_target


if __name__ == "__main__":
    a,b,c = load_loan()
    print("Yay!")