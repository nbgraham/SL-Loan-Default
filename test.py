from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from analysis import analyze
from decision_tree import DecisionTreeClassifier as MyTree, Attribute
from random_forest import RandomForestClassifier


def compare(iris, max_depth=100, f='gini', min_samples=1):
    clf = DecisionTreeClassifier(random_state=0, max_depth=max_depth, criterion=f, min_samples_split=min_samples)

    clf.fit(iris.data, iris.target)
    a = clf.predict_proba(iris.data[94].reshape(1, -1))

    print(a)

    attributes = [
        Attribute("Sepal Length", False),
        Attribute("Sepal Width", False),
        Attribute("Petal Length", False),
        Attribute("Petal Width", False)
    ]

    myclf = MyTree(max_depth=max_depth, remainder_score=f, min_split_size=min_samples)
    myclf.fit(iris.data, iris.target, attributes)

    other_tree = RandomForestClassifier(max_depth=1)
    other_tree.fit(iris.data, iris.target, attributes)
    a = other_tree.predict(iris.data[50])
    b = other_tree.predict_majority(iris.data[50])

    them_bss = 0
    my_bss = 0

    me_preds = []
    them_preds = []
    for i in range(len(iris.data)):
        me = myclf.predict_prob(iris.data[i])
        them = clf.predict_proba(iris.data[i].reshape(1, -1))

        me_preds.append(me)
        them_preds.append(me)

        actual = iris.target[i]
        my_prob = [prob for (pred, prob) in me if pred == actual]
        my_prob = my_prob[0] if len(my_prob) == 1 else 0
        them_prob = them[0][actual]

        my_bss += (1-my_prob)**2
        them_bss += (1-them_prob)**2

    them_bss /= len(iris.target)
    my_bss /= len(iris.target)

    analyze(np.array(me_preds), iris.target, 0.5)

    print("Me: ", my_bss)
    print("Them: ", them_bss)

    return my_bss, them_bss


if __name__ == "__main__":
    iris_ = load_iris()

    good = True
    for j in range(8):
        for f in ['gini', 'entropy']:
            for i in range(10):
                me, them = compare(iris_,i+1, f=f, min_samples=j+2)

                if me != them:
                    print("Diff!")
                    good = False

    if good:
        print("Equal!")