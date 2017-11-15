from anytree import Node, RenderTree
import numpy as np
from scipy import stats

from information_gain import inf_a
from gini import gin_a


class Attribute:
    def __init__(self, name, categorical):
        self.name = name
        self.categorical=categorical


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_split_size=None, remainder_score='entropy'):
        self.tree = None
        self.max_depth = max_depth
        self.min_split_size = min_split_size
        self.remainder_score=remainder_score

    def fit(self,x,y, attributes):
        self.tree = self.grow_decision_tree(x,y,attributes,y[0], max_depth=self.max_depth)

        for pre, fill, node in RenderTree(self.tree):
            print("%s%s" % (pre, node.name))

    def predict_prob(self,x):
        if self.tree is None:
            return None

        c = self.tree
        while (c.children):
            for ci in c.children:
                if ci.test(x):
                    c = ci
                    break

        return c.prob

    def predict(self, x):
        probs = self.predict_prob(x)

        pred = -1
        max = 0
        for (label,prob) in probs:
            if prob > max:
                max = prob
                pred = label

        return pred

    def grow_decision_tree(self, x, y, attributes, default, max_depth=None, label_prefix="", attr_used=None):
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
                    subtree.prob = [(y_vals[i], counts[i] / len(examples_y)) for i in range(len(y_vals))]
                    subtree.parent = tree
            else:
                for f in [("<=",lambda a,b: a<=b), (">",lambda a,b: a>b)]:
                    examples_x = np.vstack([x[i] for i in range(len(x)) if f[1](x[i,best_attribute_i], best_split_point)])
                    examples_y = np.hstack([y[i] for i in range(len(x)) if f[1](x[i,best_attribute_i], best_split_point)])

                    label = stats.mode(examples_y).mode[0]
                    subtree = self.grow_decision_tree(examples_x, examples_y, attributes, label, max_depth=next_depth, label_prefix=f[0] + str(best_split_point) + ". ", attr_used=attr_used)
                    subtree.test = reg_test(f, best_attribute_i, best_split_point)

                    y_vals, counts = np.unique(examples_y, return_counts=True)
                    subtree.prob = [(y_vals[i], counts[i]/len(examples_y)) for i in range(len(y_vals))]
                    subtree.parent = tree

            return tree

    def choose_best_attribute(self, attributes, attr_used, x, y, categorical=True):
        min = 10 ** 10
        best_attr_i = -1
        best_split_point = -1

        if self.remainder_score == 'entropy':
            score = inf_a
        elif self.remainder_score == 'gini':
            score = gin_a

        for attr_i in range(len(attributes)):
            sum = 0
            if attributes[attr_i].categorical:
                if attr_used[attr_i]:
                    continue

                vals, indices = np.unique(x[:, attr_i], return_inverse=True)

                for j in range(len(vals)):
                    examples_y = np.hstack([y[i] for i in range(len(x)) if indices[i] == j])

                    if categorical:
                        sum += len(examples_y) / len(y) * score(examples_y)
                    else:
                        avg_y = np.mean(examples_y)
                        for yi in examples_y:
                            sum += (yi - avg_y) ** 2

                if sum < min:
                    min = sum
                    best_attr_i = attr_i
            else:
                values = sorted(x[:, attr_i])
                for split_point in values:
                    before_split_indexes = x[:, attr_i] <= split_point
                    after_split_indexes = x[:, attr_i] > split_point

                    if np.all(before_split_indexes) or np.all(after_split_indexes):
                        continue

                    if categorical:
                        before_split_y = y[before_split_indexes]
                        after_split_y = y[after_split_indexes]

                        sum = len(before_split_y) / len(y) * score(before_split_y) + len(after_split_y) / len(
                            y) * score(after_split_y)
                    else:
                        before_split_avg = np.mean(y[before_split_indexes])
                        after_split_avg = np.mean(y[after_split_indexes])

                        sum = 0
                        for yi in y[before_split_indexes]:
                            sum += (yi - before_split_avg) ** 2
                        for yi in y[after_split_indexes]:
                            sum += (yi - after_split_avg) ** 2

                    if sum < min:
                        min = sum
                        best_attr_i = attr_i
                        best_split_point = split_point

        return best_attr_i, best_split_point


def reg_test(f, best_attr_i, split_point):
    return lambda p: f(p[best_attr_i],split_point)


def cat_test(best_attr_i, val):
    return lambda p: p[best_attr_i] == val


