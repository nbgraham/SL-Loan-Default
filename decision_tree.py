import threading
from anytree import Node, RenderTree
import numpy as np
from scipy import stats
import math
import progressbar

from information_gain import inf_a, inf_a_weighted
from gini import gin_a, gini_a_weighted


class Attribute:
    def __init__(self, name, categorical):
        self.name = name
        self.categorical=categorical


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_split_size=None, remainder_score='entropy', attr_allowed=None, show_progress=False):
        self.tree = None
        self.max_depth = max_depth
        self.min_split_size = min_split_size
        self.remainder_score=remainder_score
        self.x_shape = None
        self.attr_allowed = attr_allowed
        self.threads = None
        self.done_thread_count = 0
        self.progress_bar = None
        self.show_progress = show_progress

    def fit(self,x,y, attributes, weights = None):
        x = np.array(x)
        self.x_shape = x[0].shape

        attr_allowed = [True] * len(attributes)

        self.threads = []
        self.done_thread_count = 0

        if self.show_progress:
            total_thread_estimate = 2**(self.max_depth) if self.max_depth - len(attributes) < -10 else 2**math.ceil(((len(attributes)+9*self.max_depth)/18))
            self.progress_bar = progressbar.ProgressBar(max_value=total_thread_estimate, widgets=[
                'Building Tree ', progressbar.Percentage(), ' (', progressbar.SimpleProgress(), ') ',
                progressbar.Bar(), ' ',
                progressbar.Timer(), ' ', progressbar.ETA()
            ])
            self.progress_bar.update(0)

        self.tree = self.grow_decision_tree(Node(""), x, y, attributes, y[0], classes=len(np.unique(y)), max_depth=self.max_depth, attr_allowed=attr_allowed, weights=weights)

        for thread in self.threads:
            thread.join()

        if self.show_progress:
            self.progress_bar.finish()

    def render(self):
        for pre, fill, node in RenderTree(self.tree):
            print("%s%s" % (pre, node.name))

    def predict_prob(self, x):
        return self._predict_prob(x, self.max_depth, self.min_split_size)

    def _predict_prob(self, x, max_depth, min_split_size):
        if self.tree is None:
            raise UnboundLocalError("This object has no tree. This decision tree has not been fit, so it cannot predict.")

        x = np.array(x)
        if x[0].shape != self.x_shape:
            raise ValueError("Expected x with shape {} but got {}".format(self.x_shape, x[0].shape))

        probs = []

        for xi in x:
            c = self.tree
            depth = 0
            while c.children:
                if (max_depth is not None and depth == max_depth) or (min_split_size is not None and c.size < min_split_size):
                    break
                if depth > 100000 or (self.max_depth is not None and depth > self.max_depth):
                    raise RecursionError("Looped in tree {} times".format(depth))
                depth += 1

                match = False
                for ci in c.children:
                    if ci.test(xi):
                        c = ci
                        match=True
                        break

                if not match:
                    break

            probs.append(c.prob)

        return np.vstack(probs)

    def predict(self, x):
        probs = self.predict_prob(x)
        pred = np.argmax(probs, axis=1)
        return pred

    def _predict(self, x, max_depth, min_split_size):
        probs = self._predict_prob(x, max_depth, min_split_size)
        pred = np.argmax(probs, axis=1)
        return pred

    def grow_decision_tree(self, node, x, y, attributes, default, attr_allowed, classes=2, max_depth=None, label_prefix="", weights=None):
        node.size = len(y)

        if weights is None:
            y_vals, counts = np.unique(y, return_counts=True)
            i_counts = 0
            true_counts = []
            for i in range(classes):
                if i in y_vals:
                    true_counts.append(counts[i_counts])
                    i_counts += 1
                else:
                    true_counts.append(0)
            node.prob = np.array([true_counts[i] / len(y)
                                  for i in range(classes)]).reshape(1, -1)
        else:
            prob = np.zeros(classes)
            for label in range(classes):
                prob[label] = np.sum(weights[y==label])
            prob /= np.sum(prob)
            node.prob = prob.reshape(1,-1)

        if (len(x) == 0) or \
                (np.all([att.categorical for att in attributes]) and np.all(attr_allowed)) or \
                (max_depth is not None and max_depth < 1) or \
                (self.min_split_size is not None and len(x) < self.min_split_size):
            node.name = label_prefix + str(default)
            return node

        if len(np.unique(y)) == 1:
            node.name = label_prefix + str(y[0])
            return node

        best_attribute_i, best_split_point = self.choose_best_attribute(attributes, attr_allowed, x, y, weights)
        best_attribute = attributes[best_attribute_i]
        node.name = label_prefix + best_attribute.name

        next_depth = max_depth - 1 if max_depth is not None else None
        if best_attribute.categorical:
            attr_allowed = attr_allowed[:]
            attr_allowed[best_attribute_i] = False

            vals, indices = np.unique(x[:, best_attribute_i], return_inverse=True)
            for j in range(len(vals)):
                val = vals[j]

                indices_that_match = [i for i in range(len(x)) if indices[i] == j]
                examples_x = x[indices_that_match]
                examples_y = y[indices_that_match]
                examples_weights = None if weights is None else weights[indices_that_match]

                label = stats.mode(examples_y).mode[0]
                subtree = Node("")
                subtree.test = cat_test(best_attribute_i, val)
                subtree.parent = node

                t = threading.Thread(target=self.grow_decision_tree, args=(subtree, examples_x, examples_y, attributes, label, attr_allowed, 2, next_depth, "=" + str(val) + " || ",examples_weights))
                self.threads.append(t)
                t.start()
        else:
            for f in [("<=",lambda a,b: a<=b), (">",lambda a,b: a>b)]:
                split_indices = [i for i in range(len(x)) if f[1](x[i,best_attribute_i], best_split_point)]
                if len(split_indices) < 1:
                    break # not the best split if 0 fall on either side (it's not a split)

                examples_x = x[split_indices]
                examples_y = y[split_indices]
                examples_weights = None if weights is None else weights[split_indices]

                label = stats.mode(examples_y).mode[0]

                subtree = Node("")
                subtree.test = reg_test(f[1], best_attribute_i, best_split_point)
                subtree.parent = node

                t = threading.Thread(target=self.grow_decision_tree, args=(
                    subtree, examples_x, examples_y, attributes, label, attr_allowed, 2, next_depth, f[0] + str(best_split_point) + " || ",examples_weights))
                self.threads.append(t)
                t.start()

        self.done_thread_count += 1
        if self.show_progress:
            self.progress_bar.update(min(self.progress_bar.max_value, self.done_thread_count))
        return node

    def get_score_function(self):
        def f(y_vals, weights):
            if self.remainder_score == 'entropy':
                if weights is None:
                    return inf_a(y_vals)
                else:
                    return inf_a_weighted(y_vals, weights)
            elif self.remainder_score == 'gini':
                if weights is None:
                    return gin_a(y_vals)
                else:
                    return gini_a_weighted(y_vals, weights)
            else:
                raise ValueError("Invalid remainder_score: {}".format(self.remainder_score))
        return f

    def choose_best_attribute(self, attributes, attr_allowed, x, y, weights=None):
        min_rem = 10 ** 10
        best_attr_i = -1
        best_split_point = -1
        score = self.get_score_function()

        if self.attr_allowed is not None:
            attr_to_use_indices = np.random.choice(len(attributes), self.attr_allowed, replace=False)
            attr_allowed = [attr_allowed[i] and i in attr_to_use_indices for i in range(len(attributes))]

        for attr_i in range(len(attributes)):
            if not attr_allowed[attr_i]:
                continue

            rem = 0
            if attributes[attr_i].categorical:
                vals, indices = np.unique(x[:, attr_i], return_inverse=True)

                for j in range(len(vals)):
                    split_indices = [i for i in range(len(x)) if indices[i] == j]
                    examples_y = y[split_indices]
                    examples_weights = weights[split_indices] if weights is not None else None
                    rem += len(examples_y) / len(y) * score(examples_y, examples_weights)
                    if rem > min_rem:
                        break

                if rem < min_rem:
                    min_rem = rem
                    best_attr_i = attr_i
            else:
                values = sorted(np.unique(x[:, attr_i]))
                for i in range(0, len(values), math.ceil(len(values)/100)):
                    split_point = values[i]

                    before_split_indexes = x[:, attr_i] <= split_point
                    after_split_indexes = x[:, attr_i] > split_point

                    if np.all(before_split_indexes) or np.all(after_split_indexes):
                        continue

                    before_split_y = y[before_split_indexes]
                    before_weights = weights[before_split_indexes] if weights is not None else None
                    after_split_y = y[after_split_indexes]
                    after_weights = weights[after_split_indexes] if weights is not None else None

                    if len(before_split_y) < len(after_split_y):
                        first = (before_split_y, before_weights)
                        second = (after_split_y, after_weights)
                    else:
                        first = (after_split_y, after_weights)
                        second = (before_split_y, before_weights)

                    rem = len(first[0]) / len(y) * score(*first)
                    if rem > min_rem:
                        continue
                    rem += len(second[0]) / len(y) * score(*second)

                    if rem < min_rem:
                        min_rem = rem
                        best_attr_i = attr_i
                        best_split_point = split_point

        return best_attr_i, best_split_point


class DecisionTreeRegressor(DecisionTreeClassifier):
    def choose_best_attribute(self, attributes, attr_allowed, x, y):
        min_rem = 10 ** 10
        best_attr_i = -1
        best_split_point = -1
        score = self.get_score_function()

        for attr_i in range(len(attributes)):
            rem = 0
            if attributes[attr_i].categorical:
                if attr_allowed[attr_i]:
                    continue

                vals, indices = np.unique(x[:, attr_i], return_inverse=True)

                for j in range(len(vals)):
                    examples_y = np.hstack([y[i] for i in range(len(x)) if indices[i] == j])
                    avg_y = np.mean(examples_y)
                    for yi in examples_y:
                        rem += (yi - avg_y) ** 2

                if rem < min_rem:
                    min_rem = rem
                    best_attr_i = attr_i
            else:
                values = sorted(x[:, attr_i])
                for split_point in values:
                    before_split_indexes = x[:, attr_i] <= split_point
                    after_split_indexes = x[:, attr_i] > split_point

                    if np.all(before_split_indexes) or np.all(after_split_indexes):
                        # split leaves on side empty
                        continue

                    before_split_avg = np.mean(y[before_split_indexes])
                    after_split_avg = np.mean(y[after_split_indexes])

                    rem = 0
                    for yi in y[before_split_indexes]:
                        rem += (yi - before_split_avg) ** 2
                    for yi in y[after_split_indexes]:
                        rem += (yi - after_split_avg) ** 2

                    if rem < min_rem:
                        min_rem = rem
                        best_attr_i = attr_i
                        best_split_point = split_point

        return best_attr_i, best_split_point


def reg_test(f, best_attr_i, split_point):
    return lambda p: f(p[best_attr_i],split_point)


def cat_test(best_attr_i, val):
    return lambda p: p[best_attr_i] == val


