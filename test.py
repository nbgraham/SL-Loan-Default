from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from analysis import analyze
from decision_tree import DecisionTreeClassifier as MyTree, Attribute
from random_forest import RandomForestClassifier


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
                same = compare_decision_trees(data, attributes, max_depth, f=f, min_samples=min_sm)
                print("Run {}/{}".format(i,total))
                i += 1
                if not same:
                    print("------------ Diff! ----------")
                    good = False

    if good:
        print("Equal!")

def compare_random_forests(iris, n_trees=10, max_depth=100, f='gini', min_samples=1):
    other_tree = RandomForestClassifier(n_trees=n_trees, max_depth=max_depth, remainder_score=f, min_samples=min_samples)
    other_tree.fit(iris.data, iris.target, attributes)
    a = other_tree.predict(iris.data[50])
    b = other_tree.predict_majority(iris.data[50])

def compare_decision_trees(iris, attributes, max_depth=100, f='gini', min_samples=1):
    clf = DecisionTreeClassifier(random_state=0, max_depth=max_depth, criterion=f, min_samples_split=min_samples)

    clf.fit(iris.data, iris.target)
    a = clf.predict_proba(iris.data[94].reshape(1, -1))

    myclf = MyTree(max_depth=max_depth, remainder_score=f, min_split_size=min_samples)
    myclf.fit(iris.data, iris.target, attributes)

    them_bss = 0
    my_bss = 0

    me_preds = []
    them_preds = []
    for i in range(len(iris.data)):
        me = myclf.predict_prob(iris.data[i])
        them = clf.predict_proba(iris.data[i].reshape(1, -1))

        my_pred_happened = 0
        for (label, pred) in me:
            if label == 1:
                my_pred_happened=pred
        me_preds.append(my_pred_happened)

        them_preds.append(them[0][1])

        actual = iris.target[i]
        my_prob = [prob for (pred, prob) in me if pred == actual]
        my_prob = my_prob[0] if len(my_prob) == 1 else 0
        them_prob = them[0][actual]

        my_bss += (1-my_prob)**2
        them_bss += (1-them_prob)**2

    them_bss /= len(iris.target)
    my_bss /= len(iris.target)

    me_acc, me_pod, me_pofd = analyze(np.array(me_preds), iris.target, 0.5)
    them_acc, them_pod, them_pofd = analyze(np.array(them_preds), iris.target, 0.5)

    my_bss = round(my_bss,5)
    me_acc = round(me_acc, 5)
    me_pod = round(me_pod, 5)
    me_pofd = round(me_pofd, 5)

    them_bss = round(them_bss, 5)
    them_acc = round(them_acc, 5)
    them_pod = round(them_pod, 5)
    them_pofd = round(them_pofd, 5)

    print("Me:   {} {} {} {}".format(my_bss, me_acc, me_pod, me_pofd))
    print("Them: {} {} {} {}".format(them_bss, them_acc, them_pod, them_pofd))

    return my_bss == them_bss and me_acc == them_acc and me_pod == them_pod and me_pofd == them_pofd


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

    test_decision_tree(iris_, attributes)
