from anytree import Node, RenderTree
import numpy as np
from scipy import stats
from math import log2


class Attribute:
    def __init__(self, name, categorical):
        self.name = name
        self.categorical=categorical
        self.used = False


def grow_decision_tree(x, y, attributes, default, label_prefix=""):
    if len(x) == 0:
        return Node(label_prefix + str(default))
    elif len(attributes) == 0:
        return Node(label_prefix + str(default))

    if len(np.unique(y)) == 1:
        return Node(label_prefix + str(y[0]))

    best_attribute_i, best_split_point = choose_best_attribute(attributes, x, y)
    best_attribute = attributes[best_attribute_i]
    best_attribute.used = True
    tree = Node(label_prefix + best_attribute.name)

    if best_attribute.categorical:

        vals, indices = np.unique(x[:, best_attribute_i], return_inverse=True)
        for j in range(len(vals)):
            val = vals[j]
            examples_x = np.vstack([x[i] for i in range(len(x)) if indices[i] == j])
            examples_y = np.hstack([y[i] for i in range(len(x)) if indices[i] == j])

            label = stats.mode(examples_y).mode[0]
            subtree = grow_decision_tree(examples_x, examples_y, attributes, label, label_prefix="=" + str(val) + ". ")
            subtree.parent = tree
    else:
        for f in [("<=",lambda a,b: a<=b), (">",lambda a,b: a>b)]:
            examples_x = np.vstack([x[i] for i in range(len(x)) if f[1](x[i,best_attribute_i], best_split_point)])
            examples_y = np.hstack([y[i] for i in range(len(x)) if f[1](x[i,best_attribute_i], best_split_point)])

            label = stats.mode(examples_y).mode[0]
            subtree = grow_decision_tree(examples_x, examples_y, attributes, label, label_prefix=f[0] + str(best_split_point) + ". ")
            subtree.parent = tree

    return tree


def choose_best_attribute(attributes, x, y, categorical=True):
    min = 10**10
    max = 0
    best_attr_i = -1
    best_split_point = -1

    for attr_i in range(len(attributes)):
        sum = 0
        rem = 0
        if attributes[attr_i].categorical:
            if attributes[attr_i].used:
                continue

            vals, indices = np.unique(x[:, attr_i], return_inverse=True)

            for j in range(len(vals)):
                examples_y = np.hstack([y[i] for i in range(len(x)) if indices[i] == j])

                if categorical:
                    rem += len(examples_y)/len(y)*inf_a(examples_y)

                    if rem > max:
                        max = rem
                        best_attr_i = attr_i
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

                    rem = len(before_split_y)/len(y)*inf_a(before_split_y) + len(after_split_y)/len(y)*inf_a(after_split_y)

                    if rem > max:
                        max = rem
                        best_attr_i = attr_i
                        best_split_point = split_point
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


def inf_a(array):
    vals, counts = np.unique(array, return_counts=True)
    fractions = [i/len(array) for i in counts]
    return inf(*fractions)


def inf(*args):
    sum = 0
    for a in args:
        sum += -1*a*log2(a)
    return sum


if __name__ == "__main__":
    # data = np.array([[1,83, 85],[1,85,58],[0,102,102],[0,101,99]])
    data = np.array(([
        [6,1,4],
        [5,0,2],
        [3,1,-3],
        [2,0,3],
        [1,0,1]
    ]))

    x = data[:,0:2]
    y = data[:,2]
    attributes = [Attribute("A", False),Attribute("B", True)]

    tree = grow_decision_tree(x,y,attributes,0)
    for pre, fill, node in RenderTree(tree):
        print("%s%s" % (pre, node.name))