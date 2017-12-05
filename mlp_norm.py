from sklearn.neural_network import MLPClassifier
import numpy as np


class MLP_Norm():
    def __init__(self):
        self.model = MLPClassifier()
        self.maxs = None

    def fit(self, x, y):
        self.maxs = np.max(x, axis=0)
        self.model.fit(x/self.maxs,y)

    def predict_proba(self, x):
        return self.model.predict_proba(x/self.maxs)