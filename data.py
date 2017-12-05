import csv
import numpy as np
import math

from decision_tree import Attribute

np.random.seed(1)


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

def load_loan():
    with open(r'UCI_Credit_Card.csv','r') as csvfile:
    # with open(r'/home/nick/Downloads/default of credit card clients.csv','r') as csvfile:
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

    return data, target, attributes


def load_loan_no_history():
    with open(r'UCI_Credit_Card.csv','r') as csvfile:
    # with open(r'/home/nick/Downloads/default of credit card clients.csv','r') as csvfile:
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


    return data, target, attributes[:5]


def load_loan_avg_time():
    with open(r'UCI_Credit_Card.csv','r') as csvfile:
    # with open(r'/home/nick/Downloads/default of credit card clients.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        next(reader, None)

        rows = []
        targets = []
        for row in reader:
            # Exclude ID column
            avg_repayment = 0
            for i in range(6,11):
                avg_repayment += int(row[i])
            avg_repayment /= 5
            row_data = row[1:-1]
            row_data.append(avg_repayment)

            rows.append(np.array(row_data, dtype=int))
            targets.append(row[-1])

    data = np.vstack(rows)
    target = np.array(targets, dtype=int)

    at = attributes[:]
    at.append(Attribute("Average Months Late", False))

    return data, target, at

def split(data, target, training=0.8):
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    split_point = math.ceil(training*len(data))

    training_data = data[indices[:split_point]]
    training_target = target[indices[:split_point]]

    test_data = data[indices[split_point:]]
    test_target = target[indices[split_point:]]

    return training_data, training_target, test_data, test_target


def custom_sample(data, target, majority_sample_rate, minority_sample_rate):
    n_minority = np.sum(target)
    n_majority = len(target) - n_minority

    return _sample(data, target, math.floor(n_minority*minority_sample_rate), math.floor(n_majority*majority_sample_rate))


def _sample(data, target, n_minority_samples, n_majority_samples):
    majority = target == 0
    minority = target == 1

    minority_indices = np.arange(len(data))[minority]
    minority_samples = np.random.choice(minority_indices, n_minority_samples)

    majority_indices = np.arange(len(data))[majority]
    majority_samples = np.random.choice(majority_indices, n_majority_samples)

    all = np.hstack([twod(data), twod(target)])
    minority_instances = all[minority_samples]
    majority_instances = all[majority_samples]

    sample = np.vstack([minority_instances, majority_instances])
    np.random.shuffle(sample)
    return sample[:,:-1], sample[:,-1]


def twod(data):
    if len(data.shape) == 1:
        return data.reshape(-1,1)
    return data


def balanced_sampling(data, target, n_samples):
    return _sample(data, target, math.floor(n_samples/2), math.ceil(n_samples/2))


if __name__ == "__main__":
    a,b,c = load_loan()

    data, target = custom_sample(a,b,0.1,0.2)

    print("Yay!")