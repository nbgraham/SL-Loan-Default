import math
import numpy as np
import threading
import progressbar

from decision_tree import DecisionTreeClassifier

np.random.seed(1)


def bagging(x,y,i,n):
    data_to_use_indices = np.random.choice(len(x), len(x))
    return x[data_to_use_indices], y[data_to_use_indices]


class RandomForestClassifier:
    def __init__(self, n_trees=10, max_features=lambda x: math.floor(math.sqrt(x)), min_split_size=None, max_depth=None, remainder_score='gini', show_progress=False, sample=bagging):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.remainder_score = remainder_score
        self.min_split_size = min_split_size
        self.show_progress = show_progress
        self.sample = sample

        self.trees = None

    def fit(self,x,y, attributes):
        x = np.array(x)
        self.x_shape = x[0].shape

        if self.show_progress:
            self.progress_bar = progressbar.ProgressBar(max_value=self.n_trees, widgets=[
                'Building Forest ', progressbar.Percentage(), ' (', progressbar.SimpleProgress(), ') ',
                progressbar.Bar(),
                progressbar.Timer(), ' ', progressbar.ETA()
            ])
            self.progress_bar.update(0)

        self.trees = []
        threads = []
        for i in range(self.n_trees):
            t = threading.Thread(target=self.fit_one_tree, args=(x, y, attributes, i))
            threads.append(t)
            t.start()

        for thread in threads:
            thread.join()

    def fit_one_tree(self, x, y, attributes, i):
        tree = DecisionTreeClassifier(min_split_size=self.min_split_size, max_depth=self.max_depth,
                                      remainder_score=self.remainder_score,
                                      attr_allowed=self.max_features(len(attributes)), show_progress=True)

        samp_x, samp_y = self.sample(x, y, i, self.n_trees)
        tree.fit(samp_x, samp_y, attributes)

        self.trees.append(tree)
        if self.show_progress:
            self.progress_bar.update(len(self.trees))

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
                average_probs = probs
            else:
                average_probs += probs
        average_probs /= self.n_trees

        return average_probs

    def predict(self, x):
        probs = self.predict_prob(x)

        pred = np.argmax(probs, axis=1)

        return pred

    def predict_prob_vote(self, x):
        # Assume binary classification
        result = np.zeros((len(x),2))

        for tree in self.trees:
            pred = tree.predict(x)

            for i in range(len(pred)):
                result[i][pred[i]] += 1

        result = result / self.n_trees

        return result

    def predict_majority(self, x):
        label_counts = []
        for i in range(len(x)):
            label_counts.append({})

        for tree in self.trees:
            pred = tree.predict(x)

            for i in range(len(pred)):
                if pred[i] in label_counts[i]:
                    label_counts[i][pred[i]] += 1
                else:
                    label_counts[i][pred[i]] = 1

        winners = []
        for ct in label_counts:
            max = 0
            pred = None
            for label, count in ct.items():
                if count > max:
                    max = count
                    pred = label
            winners.append(pred)

        return np.array(winners)