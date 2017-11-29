import numpy as np


class KnnClassifier():
    def __init__(self, k, weights=None):
        self.k = k
        self.weights = weights
        self.normalized_data = None
        self.normalizer = None
        self.target = None
        self.attributes = None

    def train(self,x,y,attributes=None):
        if self.weights is None:
            self.weights = np.ones(x[0].shape)

        if x[0].shape != self.weights.shape:
            raise ValueError("Expected x's with shape {} but got shape{}", self.weights.shape, x[0].shape)

        self.normalizer = x.max(axis=0)
        self.normalized_data = x / self.normalizer
        self.target = y
        self.attributes = attributes

    def predict_prob(self, x):
        normalized_xs = x / self.normalizer
        for xi in normalized_xs:
            neighbors = []
            neighbors_diffs = []

            for instance in self.normalized_data:
                diff = (instance-xi)*self.weights
                ssd = np.sum(diff*2)

                if len(neighbors) < self.k:
                    i = 0
                    if len(neighbors_diffs) < 1:
                        neighbors_diffs.append(ssd)
                    else:
                        while neighbors_diffs[i] < ssd:
                            i += 1
                    neighbors_diffs.insert(i, ssd)
                    neighbors.insert(i, instance)
                elif ssd < neighbors_diffs[-1]:
                    i = 0
                    while neighbors_diffs[i] < ssd:
                        i += 1
                    neighbors_diffs[i] = ssd
                    neighbors[i] = instance

        # avg neighbors (with distances?)


