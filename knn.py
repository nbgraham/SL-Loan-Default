import numpy as np

class KnnClassifier():
    def __init__(self, k=1, weights=None):
        self.k = k
        self.weights = weights
        self.normalized_data = None
        self.normalizer = None
        self.target = None
        self.attributes = None

    def fit(self,x,y,attributes=None):
        if self.weights == 1 or self.weights is None:
            self.weights = np.ones(x[0].shape)

        if x[0].shape != self.weights.shape:
            raise ValueError("Expected x's with shape {} but got shape{}", self.weights.shape, x[0].shape)

        self.normalizer = x.max(axis=0)
        self.normalized_data = x / self.normalizer
        self.target = y
        self.attributes = attributes

    def predict_prob(self, x):
        normalized_xs = x / self.normalizer

        results = []
        for xi in normalized_xs:
            neighbors = []
            neighbors_diffs = []

            for j in range(len(self.normalized_data)):
                diff = (self.normalized_data[j]-xi)*self.weights
                ssd = np.sum(diff**2)

                if len(neighbors) < self.k:
                    i = 0
                    if len(neighbors_diffs) < self.k:
                        neighbors_diffs.append(ssd)
                    else:
                        while i < self.k and neighbors_diffs[i] < ssd:
                            i += 1
                            print(i,len(neighbors_diffs))
                        neighbors_diffs.insert(i, ssd)
                    neighbors.insert(i, j)
                elif ssd < neighbors_diffs[-1]:
                    i = 0
                    while i < self.k and neighbors_diffs[i] < ssd:
                        i += 1
                    neighbors_diffs.insert(i,ssd)
                    neighbors.insert(i,j)
            neighbors = np.array([self.target[j] for j in neighbors])
            neighbors_diffs = np.array(neighbors_diffs)
            avg = np.sum(neighbors*neighbors_diffs)/np.sum(neighbors_diffs)
            results.append(avg)

        results = np.array(results)
        prob0 = np.ones(results.shape)-results
        pred = np.hstack([prob0.reshape(-1,1),results.reshape(-1,1)])
        return pred




