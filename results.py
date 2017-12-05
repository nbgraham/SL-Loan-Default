from sklearn.tree import DecisionTreeClassifier as SKLearnDecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as SKLearnRandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier as SKLearnKNeighborsClassifier
from matplotlib import pyplot as plt

from analysis import test_f_stars, rel, inv_rel
from data import load_loan, split
from random_forest import RandomForestClassifier as MyRandomForestClassifier
from decision_tree import DecisionTreeClassifier as MyDecisionTreeClassifier
from adaboost import Adaboost as MyAdaboostClassifier
from knn import KnnClassifier as MyKnnClassifier
from wrapper import Wrapper


def main():
    data, target, attributes = load_loan()
    training_val_data, training_val_target, test_data, test_target = split(data, target)


    dt_clf = Wrapper(my_code=True, my_model=MyDecisionTreeClassifier(max_depth=7, min_split_size=110, remainder_score='entropy'), sklearn_model=SKLearnDecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=110))
    rf_clf = Wrapper(my_code=True, my_model=MyRandomForestClassifier(max_depth=5, n_trees=70, remainder_score='gini'), sklearn_model=SKLearnRandomForestClassifier(criterion='gini', max_depth=5, n_estimators=70))
    ada_clf = Wrapper(my_code=True, my_model=MyAdaboostClassifier(max_depth=1, learning_rate=1, n_models=50),sklearn_model=AdaBoostClassifier(base_estimator=SKLearnDecisionTreeClassifier(criterion='entropy', max_depth=1), n_estimators=50,learning_rate=1))
    # clf = Wrapper(my_code=True, my_model=MyKnnClassifier(k=1, weights=None),sklearn_model=SKLearnKNeighborsClassifier(n_neighbors=1, weights='uniform'))

    f_stars = [i / 100 for i in range(100)]

    dt_clf.fit(training_val_data, training_val_target, attributes)
    dt_pred = dt_clf.predict_prob(test_data) # w/100 trees AUC = 0.78
    dt_pods, dt_pofds = test_f_stars(dt_pred, test_target, f_stars)

    rf_clf.fit(training_val_data, training_val_target, attributes)
    rf_pred = rf_clf.predict_prob(test_data)  # w/100 trees AUC = 0.78
    rf_pods, rf_pofds = test_f_stars(rf_pred, test_target, f_stars)

    ada_clf.fit(training_val_data, training_val_target, attributes)
    ada_pred = ada_clf.predict_prob(test_data)  # w/100 trees AUC = 0.78
    ada_pods, ada_pofds = test_f_stars(ada_pred, test_target, f_stars)

    plt.plot(dt_pofds, dt_pods, label="Decision Tree")
    plt.plot(rf_pofds, rf_pods, label="Random Forests")
    plt.plot(ada_pofds, ada_pods, label="Adaboost")
    plt.plot([0,1],[0,1], "--", label="x = y")

    plt.title('ROC')
    plt.xlabel('POFD')
    plt.ylabel('POD')

    plt.show()


if __name__ == "__main__":
    main()