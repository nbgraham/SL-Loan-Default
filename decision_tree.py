from anytree import Node, RenderTree
import numpy as np
from scipy import stats
from information_gain import inf_a


class Attribute:
    def __init__(self, name, categorical):
        self.name = name
        self.categorical=categorical

class DecisionTreeClassifier:
    def __init__(self):
        self.tree = None

    def fit(self,x,y, attributes):
        self.tree = grow_decision_tree(x,y,attributes,y[0])

        for pre, fill, node in RenderTree(self.tree):
            print("%s%s" % (pre, node.name))

    def predict(self,x):
        if self.tree is None:
            return None

        c = self.tree
        while (c.children):
            for ci in c.children:
                if ci.test(x):
                    c = ci
                    break

        return c.name

def grow_decision_tree(x, y, attributes, default, label_prefix="", attr_used=None):
        if attr_used is None:
            attr_used = [False]*len(attributes)

        if len(x) == 0:
            return Node(label_prefix + str(default))
        elif np.all([att.categorical for att in attributes]) and np.all(attr_used):
            return Node(label_prefix + str(default))

        if len(np.unique(y)) == 1:
            return Node(label_prefix + str(y[0]))

        best_attribute_i, best_split_point = choose_best_attribute(attributes, attr_used, x, y)
        best_attribute = attributes[best_attribute_i]
        attr_used = attr_used[:]
        attr_used[best_attribute_i] = True
        tree = Node(label_prefix + best_attribute.name)

        if best_attribute.categorical:
            vals, indices = np.unique(x[:, best_attribute_i], return_inverse=True)
            for j in range(len(vals)):
                val = vals[j]
                examples_x = np.vstack([x[i] for i in range(len(x)) if indices[i] == j])
                examples_y = np.hstack([y[i] for i in range(len(x)) if indices[i] == j])

                label = stats.mode(examples_y).mode[0]
                subtree = grow_decision_tree(examples_x, examples_y, attributes, label, label_prefix="=" + str(val) + ". ", attr_used=attr_used)
                subtree.test = cat_test(best_attribute_i, val)
                subtree.parent = tree
        else:
            for f in [("<=",lambda a,b: a<=b), (">",lambda a,b: a>b)]:
                examples_x = np.vstack([x[i] for i in range(len(x)) if f[1](x[i,best_attribute_i], best_split_point)])
                examples_y = np.hstack([y[i] for i in range(len(x)) if f[1](x[i,best_attribute_i], best_split_point)])

                label = stats.mode(examples_y).mode[0]
                subtree = grow_decision_tree(examples_x, examples_y, attributes, label, label_prefix=f[0] + str(best_split_point) + ". ", attr_used=attr_used)
                subtree.test = reg_test(f, best_attribute_i, best_split_point)
                subtree.parent = tree

        return tree


def reg_test(f, best_attr_i, split_point):
    return lambda p: f(p[best_attr_i],split_point)


def cat_test(best_attr_i, val):
    return lambda p: p[best_attr_i] == val


def choose_best_attribute(attributes, attr_used, x, y, categorical=True):
    min = 10**10
    best_attr_i = -1
    best_split_point = -1

    for attr_i in range(len(attributes)):
        sum = 0
        if attributes[attr_i].categorical:
            if attr_used[attr_i]:
                continue

            vals, indices = np.unique(x[:, attr_i], return_inverse=True)

            for j in range(len(vals)):
                examples_y = np.hstack([y[i] for i in range(len(x)) if indices[i] == j])

                if categorical:
                    sum += len(examples_y)/len(y)*inf_a(examples_y)
                else:
                    avg_y = np.mean(examples_y)
                    for yi in examples_y:
                        sum += (yi-avg_y)**2

            if sum < min:
                min = sum
                best_attr_i = attr_i
        else:
            values = sorted(x[:,attr_i])
            for split_point in values:
                before_split_indexes = x[:,attr_i]<=split_point
                after_split_indexes = x[:,attr_i]>split_point

                if np.all(before_split_indexes) or np.all(after_split_indexes):
                    continue

                if categorical:
                    before_split_y = y[before_split_indexes]
                    after_split_y = y[after_split_indexes]

                    sum = len(before_split_y)/len(y)*inf_a(before_split_y) + len(after_split_y)/len(y)*inf_a(after_split_y)
                else:
                    before_split_avg = np.mean(y[before_split_indexes])
                    after_split_avg = np.mean(y[after_split_indexes])

                    sum=0
                    for yi in y[before_split_indexes]:
                        sum += (yi-before_split_avg)**2
                    for yi in y[after_split_indexes]:
                        sum += (yi-after_split_avg)**2

                if sum < min:
                    min = sum
                    best_attr_i = attr_i
                    best_split_point = split_point

    return best_attr_i, best_split_point


