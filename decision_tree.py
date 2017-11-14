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
    tree = Node(best)

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
    return np.random.random_integers(0,len(attributes)-1)


if __name__ == "__main__":
    data = np.array([[1,83, 85],[1,85,58],[0,102,102],[0,101,99]])

    x = data[:,0:2]
    y = data[:,2]
    attributes = [0,1]

    tree = grow_decision_tree(x,y,attributes,50)
    for pre, fill, node in RenderTree(tree):
        print("%s%s" % (pre, node.name))