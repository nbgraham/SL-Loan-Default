import numpy as np
from decision_tree import Attribute, DecisionTreeClassifier

if __name__ == "__main__":
    # data = np.array([[1,83, 85],[1,85,58],[0,102,102],[0,101,99]])
    # data = np.array(([
    #     [6,1,4],
    #     [5,0,2],
    #     [3,1,-3],
    #     [2,0,3],
    #     [1,0,1]
    # ]))

    data = np.array(([
        [1, 3, 1, 0, 0],
        [1, 3, 1, 1, 0],
        [2, 3, 1, 0, 1],
        [3, 2, 1, 0, 1],
        [3, 1, 0, 0, 1],
        [3, 1, 0, 1, 0],
        [2, 1, 0, 1, 1],
        [1, 2, 1, 0, 0],
    ]))

    x = data[:,0:4]
    y = data[:,4]
    attributes = [
        Attribute("Outlook", True),
        Attribute("Temperature", True),
        Attribute("Humidity", True),
        Attribute("Windy", True)
    ]

    tree = DecisionTreeClassifier()
    tree.fit(x,y,attributes)

    new_x = [2, 3, 1, 0]
    probs = tree.predict_prob(new_x)
    pred = tree.predict(new_x)
    print(probs)
    print(pred)