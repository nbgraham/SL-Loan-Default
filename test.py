from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from decision_tree import DecisionTreeClassifier as MyTree, Attribute

if __name__ == "__main__":
    clf = DecisionTreeClassifier(random_state=0)
    iris = load_iris()

    clf.fit(iris.data, iris.target)
    a = clf.predict_proba(iris.data[94].reshape(1, -1))

    print(a)

    attributes = [
        Attribute("Sepal Length", False),
        Attribute("Sepal Width", False),
        Attribute("Petal Length", False),
        Attribute("Petal Width", False)
    ]

    myclf = MyTree()
    myclf.fit(iris.data, iris.target, attributes)

    a = myclf.predict_prob(iris.data[94])

    print(a)

