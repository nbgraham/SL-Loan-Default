import numpy as np


def gin_a(array):
    vals, counts = np.unique(array, return_counts=True)
    fractions = [i/len(array) for i in counts]
    return gin(*fractions)


def gini_a_weighted(array, weights):
    counts = {}
    for i in range(len(array)):
        val = array[i]
        if val in counts:
            counts[val] += weights[i]
        else:
            counts[val] = weights[i]

    fractions = [counts[i] / np.sum(weights) for i in counts]
    return gin(*fractions)


def gin(*args):
    sum = 0
    for a in args:
        sum += a**2
    return 1 - sum