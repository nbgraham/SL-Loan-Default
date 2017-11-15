import numpy as np


def gin_a(array):
    vals, counts = np.unique(array, return_counts=True)
    fractions = [i/len(array) for i in counts]
    return gin(*fractions)


def gin(*args):
    sum = 0
    for a in args:
        sum += a**2
    return 1 - sum