from anytree import Node, RenderTree
import numpy as np
from scipy import stats
import math

from information_gain import inf_a
from gini import gin_a


class Attribute:
    def __init__(self, name, categorical):
        self.name = name
        self.categorical=categorical


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_split_size=None, remainder_score='entropy', attr_allowed=None):
        self.tree = None
        self.max_depth = max_depth
        self.min_split_size = min_split_size
        self.remainder_score=remainder_score
        self.x_shape = None
        self.attr_allowed = attr_allowed

    def fit(self,x,y, attributes):
        x = np.array(x)
        self.x_shape = x[0].shape

        if self.attr_allowed is None:
            attr_used = [False] * len(attributes)
        else:
            attr_to_use_indices = np.random.choice(len(attributes), self.attr_allowed, replace=False)
            attr_used = [i not in attr_to_use_indices for i in range(len(attributes))]

        self.tree = self.grow_decision_tree(x,y,attributes,y[0], classes=len(np.unique(y)), max_depth=self.max_depth, attr_used=attr_used)

    def render(self):
        for pre, fill, node in RenderTree(self.tree):
            print("%s%s" % (pre, node.name))

    def predict_prob(self,x):
        if self.tree is None:
            raise UnboundLocalError("This object has no tree. This decision tree has not been fit, so it cannot predict.")

        x = np.array(x)
        if x[0].shape != self.x_shape:
            raise ValueError("Expected x with shape {} but got {}".format(self.x_shape, x[0].shape))

        probs = []

        for xi in x:
            c = self.tree
            while c.children:
                for ci in c.children:
                    if ci.test(xi):
                        c = ci
                        break
            probs.append(c.prob)

        return np.vstack(probs)

    def predict(self, x):
        probs = self.predict_prob(x)

        pred = np.argmax(probs, axis=1)

        return pred

    def grow_decision_tree(self, x, y, attributes, default, classes=2, max_depth=None, label_prefix="", attr_used=None):
            if attr_used is None:
                attr_used = [False]*len(attributes)

            if (len(x) == 0) or \
                (np.all([att.categorical for att in attributes]) and np.all(attr_used)) or \
                (max_depth is not None and max_depth < 1) or \
                (self.min_split_size is not None and len(x) < self.min_split_size):
                return Node(label_prefix + str(default))

            if len(np.unique(y)) == 1:
                return Node(label_prefix + str(y[0]))

            best_attribute_i, best_split_point = self.choose_best_attribute(attributes, attr_used, x, y)
            best_attribute = attributes[best_attribute_i]
            attr_used = attr_used[:]
            attr_used[best_attribute_i] = True
            tree = Node(label_prefix + best_attribute.name)

            next_depth = max_depth - 1 if max_depth is not None else None
            if best_attribute.categorical:
                vals, indices = np.unique(x[:, best_attribute_i], return_inverse=True)
                for j in range(len(vals)):
                    val = vals[j]
                    examples_x = np.vstack([x[i] for i in range(len(x)) if indices[i] == j])
                    examples_y = np.hstack([y[i] for i in range(len(x)) if indices[i] == j])

                    label = stats.mode(examples_y).mode[0]
                    subtree = self.grow_decision_tree(examples_x, examples_y, attributes, label, max_depth=next_depth, label_prefix="=" + str(val) + ". ", attr_used=attr_used)
                    subtree.test = cat_test(best_attribute_i, val)

                    y_vals, counts = np.unique(examples_y, return_counts=True)

                    i_counts = 0
                    true_counts = []
                    for i in range(classes):
                        if i in y_vals:
                            true_counts.append(counts[i_counts])
                            i_counts += 1
                        else:
                            true_counts.append(0)

                    subtree.prob = np.array([true_counts[i] / len(examples_y)
                                             for i in range(classes)]).reshape(1,-1)
                    subtree.parent = tree
            else:
                for f in [("<=",lambda a,b: a<=b), (">",lambda a,b: a>b)]:
                    examples_x = np.vstack([x[i] for i in range(len(x)) if f[1](x[i,best_attribute_i], best_split_point)])
                    examples_y = np.hstack([y[i] for i in range(len(x)) if f[1](x[i,best_attribute_i], best_split_point)])

                    label = stats.mode(examples_y).mode[0]
                    subtree = self.grow_decision_tree(examples_x, examples_y, attributes, label, max_depth=next_depth, label_prefix=f[0] + str(best_split_point) + ". ", attr_used=attr_used)
                    subtree.test = reg_test(f[1], best_attribute_i, best_split_point)

                    y_vals, counts = np.unique(examples_y, return_counts=True)

                    i_counts = 0
                    true_counts = []
                    for i in range(classes):
                        if i in y_vals:
                            true_counts.append(counts[i_counts])
                            i_counts += 1
                        else:
                            true_counts.append(0)

                    subtree.prob = np.array([true_counts[i] / len(examples_y)
                                             for i in range(classes)]).reshape(1,-1)
                    subtree.parent = tree

            return tree

    def get_score_function(self):
        if self.remainder_score == 'entropy':
            score = inf_a
        elif self.remainder_score == 'gini':
            score = gin_a
        else:
            raise ValueError("Invalid remainder_score: {}".format(self.remainder_score))
        return score

    def choose_best_attribute(self, attributes, attr_used, x, y):
        min_rem = 10 ** 10
        best_attr_i = -1
        best_split_point = -1
        score = self.get_score_function()

        for attr_i in range(len(attributes)):
            rem = 0
            if attributes[attr_i].categorical:
                if attr_used[attr_i]:
                    continue

                vals, indices = np.unique(x[:, attr_i], return_inverse=True)

                for j in range(len(vals)):
                    examples_y = np.hstack([y[i] for i in range(len(x)) if indices[i] == j])
                    rem += len(examples_y) / len(y) * score(examples_y)

                if rem < min_rem:
                    min_rem = rem
                    best_attr_i = attr_i
            else:
                values = sorted(x[:, attr_i])
                for i in range(0,len(values),math.ceil(len(values)/100)):
                    split_point = values[i]

                    before_split_indexes = x[:, attr_i] <= split_point
                    after_split_indexes = x[:, attr_i] > split_point

                    if np.all(before_split_indexes) or np.all(after_split_indexes):
                        continue

                    before_split_y = y[before_split_indexes]
                    after_split_y = y[after_split_indexes]
                    rem = len(before_split_y) / len(y) * score(before_split_y) \
                          + len(after_split_y) / len(y) * score(after_split_y)

                    if rem < min_rem:
                        min_rem = rem
                        best_attr_i = attr_i
                        best_split_point = split_point

        return best_attr_i, best_split_point


class DecisionTreeRegressor(DecisionTreeClassifier):
    def choose_best_attribute(self, attributes, attr_used, x, y):
        min_rem = 10 ** 10
        best_attr_i = -1
        best_split_point = -1
        score = self.get_score_function()

        for attr_i in range(len(attributes)):
            rem = 0
            if attributes[attr_i].categorical:
                if attr_used[attr_i]:
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


