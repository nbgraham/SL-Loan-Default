import math
import numpy as np

from decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(self, n_trees=10, max_features=lambda x: math.floor(math.sqrt(x)), min_split_size=None, max_depth=None, remainder_score='gini'):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.remainder_score = remainder_score
        self.min_split_size = min_split_size

        self.trees = None

    def fit(self,x,y, attributes):
        x = np.array(x)
        self.x_shape = x[0].shape

        self.trees = []
        for i in range(self.n_trees):
            tree = DecisionTreeClassifier(min_split_size=self.min_split_size, max_depth=self.max_depth, remainder_score=self.remainder_score, attr_allowed=self.max_features(len(attributes)))

            data_to_used_indices = np.random.choice(len(x), len(x))
            tree.fit(x[data_to_used_indices], y[data_to_used_indices], attributes)

            self.trees.append(tree)

    def predict_prob(self,x):
        if self.trees is None:
            raise UnboundLocalError("This object has no trees. This random forest has not been fit, so it cannot predict.")

        x = np.array(x)
        if x[0].shape != self.x_shape:
            raise ValueError("Expected x with shape {} but got {}".format(self.x_shape, x[0].shape))

        average_probs = None
        for classifier in self.trees:
            probs = classifier.predict_prob(x)
            if average_probs is None:
                average_probs = probs/self.n_trees
            else:
                average_probs += probs/self.n_trees

        return average_probs

    def predict(self, x):
        probs = self.predict_prob(x)

        pred = np.argmax(probs, axis=1)

        return pred

    def predict_majority(self, x):
        label_counts = {}

        for tree in self.trees:
            pred = tree.predict(x)

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


