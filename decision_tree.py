from anytree import Node, RenderTree
import numpy as np
from scipy import stats


def grow_decision_tree(x, y, attributes, default):
    if len(x) == 0:
        return Node(default)
    elif len(attributes) == 0:
        return Node(default)

    if len(np.unique(y)) == 1:
        return Node(y[0])

    best = choose_best_attribute(attributes, x, y)
    tree = Node(attributes[best])

    vals, indices = np.unique(x[:,best], return_inverse=True)
    for j in range(len(vals)):
        val = vals[j]
        examples_x = np.vstack([x[i] for i in range(len(x)) if indices[i] ==  j])
        examples_y = np.hstack([y[i] for i in range(len(x)) if indices[i] ==  j])

        label = stats.mode(examples_y).mode[0]
        subtree = grow_decision_tree(examples_x, examples_y, attributes, label)
        subtree.parent = tree

    return tree



def choose_best_attribute(attributes, x, y):
    min = 10**10
    best_attr_i = -1
    best_split_point = -1

    for attr_i in range(len(attributes)):
        values = sorted(x[:,attr_i])
        for split_point in [(values[i] + values[i+1])/2 for i in range(len(values)-1)]:
            before_split_indexes = x[:,attr_i]<=split_point
            after_split_indexes = x[:,attr_i]>split_point

            before_split_avg = np.mean(y[before_split_indexes])
            after_split_avg = np.mean(y[after_split_indexes])

            sum = 0
            for yi in y[before_split_indexes]:
                sum += (yi-before_split_avg)**2
            for yi in y[after_split_indexes]:
                sum += (yi-after_split_avg)**2

            if sum < min:
                min = sum
                best_attr_i = attr_i
                best_split_point = split_point

    return best_attr_i


if __name__ == "__main__":
    data = np.array([[1,83, 85],[1,85,58],[0,102,102],[0,101,99]])

    x = data[:,0:2]
    y = data[:,2]
    attributes = ["Season","Yes. High"]

    tree = grow_decision_tree(x,y,attributes,50)
    for pre, fill, node in RenderTree(tree):
        print("%s%s" % (pre, node.name))