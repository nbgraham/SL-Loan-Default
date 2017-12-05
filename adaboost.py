import numpy as np
from math import log2
import progressbar

from decision_tree import DecisionTreeClassifier


def default_create_model(min_split_size=None, max_depth=4, remainder_score='entropy'):
    return DecisionTreeClassifier(min_split_size=min_split_size, max_depth=max_depth,
                       remainder_score=remainder_score, show_progress=False)


class Adaboost:
    def __init__(self,max_depth=4, n_models=10, learning_rate = 1, create_model=default_create_model, show_progress=True):
        self.n_models = n_models
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.create_model = create_model
        self.models = []
        self.errs = []
        self.show_progress = show_progress
        self.progress_bar = None

    def fit(self, x, y, attributes):
        self.models = []
        self.errs = []

        # initialize weights to 1 initially
        weights = np.ones(len(x))/len(x)

        if self.show_progress:
            self.progress_bar = progressbar.ProgressBar(max_value=self.n_models, widgets=[
                'Building Adaboost classifier ', progressbar.Percentage(), ' (', progressbar.SimpleProgress(), ') ',
                progressbar.Bar(),
                progressbar.Timer(), ' ', progressbar.ETA()
            ])

        for i in range(self.n_models):
            if self.show_progress:
                self.progress_bar.update(i)
            # create a default decision tree with max depth then fit with current weights
            model = self.create_model(max_depth=self.max_depth)
            model.fit(x, y, attributes, weights=weights)
            predictions = model.predict(x)

            err_sum = 0
            weight_change = np.zeros(len(x))
            for i in range(len(y)):
                if predictions[i] != y[i]:
                    err_sum += weights[i]
                    weight_change[i] = 1

            if err_sum == 0 or err_sum > 0.5:
                break

            self.models.append(model)
            self.errs.append(err_sum)

            weight_change[weight_change == 0] = err_sum / (1 - err_sum) * self.learning_rate
            weights *= weight_change
            weights /= np.sum(weights)

        return self.models, self.errs

    def predict_prob(self,x):
        # Assume binary
        classification = np.zeros((2,len(x)))

        for i in range(self.n_models):
            if i >= len(self.models):
                break

            preds = self.models[i].predict(x)

            for pred in np.unique(preds):
                classification[pred][preds==pred] -= log2(self.errs[i]/(1-self.errs[i]))

        classification = classification.T
        classification /= np.sum(classification[0])

        return classification

    def predict(self,x):
        predictions = self.predict_prob(x)

        return np.argmax(predictions, axis=1)

