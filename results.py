from sklearn.tree import DecisionTreeClassifier as SKLearnDecisionTreeClassifier

from analysis import test_f_stars, rel, inv_rel
from data import load_loan, split
from random_forest import RandomForestClassifier
from decision_tree import DecisionTreeClassifier as MyDecisionTreeClassifier
from wrapper import Wrapper


def main():
    data, target, attributes = load_loan()
    training_val_data, training_val_target, test_data, test_target = split(data, target)

    clf = Wrapper(my_code=False, my_model=MyDecisionTreeClassifier(max_depth=7, min_split_size=110, remainder_score='entropy'), sklearn_model=SKLearnDecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=110))

    clf.fit(training_val_data, training_val_target, attributes)

    pred = clf.predict_prob(test_data) # w/100 trees AUC = 0.78

    # inv_rel(pred[:, 1], test_target)
    rel(pred[:, 1], test_target, plot=True)

    f_stars = [i / 100 for i in range(100)]
    auc, max_acc, pod, pofd = test_f_stars(pred, test_target, f_stars, plot=True)
    print("\nAUC: ", auc, " Max Accuracy: ", max_acc, " Min Square ROC dist: POD: ", pod, " POFD: ", pofd)

if __name__ == "__main__":
    main()