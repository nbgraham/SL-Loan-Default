import numpy as np
from math import log2


def inf_a(array):
    vals, counts = np.unique(array, return_counts=True)
    fractions = [i/len(array) for i in counts]
    return inf(*fractions)


def inf(*args):
    sum = 0
    for a in args:
        sum += -1*a*log2(a)
    return sum