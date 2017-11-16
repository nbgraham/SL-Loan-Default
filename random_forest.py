import math
import numpy as np

from decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(self, n_trees=10, max_features=lambda x: math.floor(math.sqrt(x)), max_depth=None, remainder_score='gini'):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.remainder_score = remainder_score
        self.trees = None

    def fit(self,x,y, attributes):
        x = np.array(x)
        self.x_shape = x[0].shape

        self.trees = []
        for i in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, remainder_score=self.remainder_score, attr_allowed=self.max_features(len(attributes)))

            data_to_used_indices = np.random.choice(len(x), len(x))
            tree.fit(x[data_to_used_indices], y[data_to_used_indices], attributes)

            self.trees.append(tree)

    def predict_prob(self,x):
        if self.trees is None:
            raise UnboundLocalError("This object has no trees. This random forest has not been fit, so it cannot predict.")

        x = np.array(x)
        if x.shape != self.x_shape:
            raise ValueError("Expected x with shape {} but got {}".format(self.x_shape, x.shape))

        probs = []
        for classifier in self.trees:
            c = classifier.tree
            while c.children:
                for ci in c.children:
                    if ci.test(x):
                        c = ci
                        break
            probs.append(c.prob)

        sum_probs = []
        for prob in probs:
            for (label, probab) in prob:
                found = False
                for i in range(len(sum_probs)):
                    (_label,_probab) = sum_probs[i]
                    if label == _label:
                        sum_probs[i] = (_label, _probab + probab/self.n_trees)
                        found = True
                        break
                if not found:
                    sum_probs.append((label,probab/self.n_trees))


        return sum_probs

    def predict(self, x):
        probs = self.predict_prob(x)

        pred = -1
        max = 0
        for (label,prob) in probs:
            if prob > max:
                max = prob
                pred = label

        return pred

    def predict_majority(self, x):
        label_counts = {}

        for tree in self.trees:
            probs = tree.predict_prob(x)

            pred = -1
            max = 0
            for (label,prob) in probs:
                if prob > max:
                    max = prob
                    pred = label

            if pred in label_counts:
                label_counts[pred] += 1
            else:
                label_counts[pred] = 1

        max = 0
        pred = None
        for label, count in label_counts.items():
            if count > max:
                max = count
                pred = label

        return pred


