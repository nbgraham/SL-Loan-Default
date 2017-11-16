from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from analysis import analyze
from decision_tree import DecisionTreeClassifier as MyTree, Attribute
from random_forest import RandomForestClassifier as MyRandom


def test_decision_tree(data, attributes):
    good = True

    _min_samples = [j+2 for j in range(8)]
    _max_depths = [i+1 for i in range(10)]
    fs = ['gini', 'entropy']

    total = len(_min_samples)*len(_max_depths)*len(fs)
    i = 0
    for min_sm in _min_samples:
        for f in fs:
            for max_depth in _max_depths:
                i += 1
                print("Run {}/{}".format(i, total))

                same = compare_decision_trees(data, attributes, max_depth, f=f, min_samples=min_sm)
                if not same:
                    print("------------ Diff! ----------")
                    good = False

    if good:
        print("Equal!")


def test_random_forests(data, attributes):
    good = True

    ns_trees = range(2,8,2)
    _max_depths = range(1,9,3)
    fs = ['gini', 'entropy']

    total = len(ns_trees)*len(_max_depths)*len(fs)
    i = 0
    for n_trees in ns_trees:
        for f in fs:
            for max_depth in _max_depths:
                i += 1
                print("Run {}/{}".format(i, total))

                same = compare_random_forests(data, attributes, max_depth=max_depth, f=f, n_trees=n_trees)
                if not same:
                    print("------------ Diff! ----------")
                    good = False

    if good:
        print("Equal!")


def compare_random_forests(iris, attributes, n_trees=10, max_depth=100, f='gini', min_samples=None):
    me = MyRandom(n_trees=n_trees, max_depth=max_depth, remainder_score=f, min_split_size=min_samples)
    me.fit(iris.data, iris.target, attributes)

    me_preds = me.predict_prob(iris.data)
    p = me.predict_majority(iris.data)
    p1 = me.predict(iris.data)

    res = compare_bss(iris.target, me_preds)
    print("BSS: ", res[0])

    return False


def compare_decision_trees(iris, attributes, max_depth=100, f='gini', min_samples=1):
    clf = DecisionTreeClassifier(random_state=0, max_depth=max_depth, criterion=f, min_samples_split=min_samples)
    clf.fit(iris.data, iris.target)

    myclf = MyTree(max_depth=max_depth, remainder_score=f, min_split_size=min_samples)
    myclf.fit(iris.data, iris.target, attributes)

    me_preds = myclf.predict_prob(iris.data)
    them_preds = clf.predict_proba(iris.data)

    # me_acc, me_pod, me_pofd = analyze(me_preds, iris.target, 0.5)
    # them_acc, them_pod, them_pofd = analyze(them_preds, iris.target, 0.5)
    #
    # res = compare_bss(iris.target, me_preds, them_preds)
    # my_bss = res[0]
    # them_bss = res[1]
    #
    # my_bss = round(my_bss,5)
    # me_acc = round(me_acc, 5)
    # me_pod = round(me_pod, 5)
    # me_pofd = round(me_pofd, 5)
    #
    # them_bss = round(them_bss, 5)
    # them_acc = round(them_acc, 5)
    # them_pod = round(them_pod, 5)
    # them_pofd = round(them_pofd, 5)
    #
    # print("Me:   {} {} {} {}".format(my_bss, me_acc, me_pod, me_pofd))
    # print("Them: {} {} {} {}".format(them_bss, them_acc, them_pod, them_pofd))

    return np.all(np.equal(np.round(me_preds,5), np.round(them_preds,5)))


def compare_bss(target, *args):
    result = [0]*len(args)
    for i in range(len(target)):
        actual = target[i]

        for j in range(len(args)):
            prob_of_actual = args[j][i][actual]
            result[j] += (1 - prob_of_actual) ** 2 / len(target)

    return result


def guess_attributes(data):
    attributes = []

    for i in range(len(data.feature_names)):
        feature = data.feature_names[i]
        x_val = data.data[0][i]

        if isinstance(x_val, float):
            attributes.append(Attribute(feature, False))
        else:
            attributes.append(Attribute(feature, True))

    return attributes


if __name__ == "__main__":
    iris_ = load_breast_cancer()

    attributes = guess_attributes(iris_)
    # attributes = [
    #     Attribute("Sepal Length", False),
    #     Attribute("Sepal Width", False),
    #     Attribute("Petal Length", False),
    #     Attribute("Petal Width", False)
    # ]

    test_random_forests(iris_, attributes)
