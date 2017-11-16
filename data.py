import csv
import numpy as np

from decision_tree import Attribute


def load_loan():
    with open(r'/home/nick/Downloads/default of credit card clients.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        next(reader, None)

        rows = []
        targets = []
        for row in reader:
            # Exclude ID column
            rows.append(np.array(row[1:-1], dtype=int))
            targets.append(row[-1])

    data = np.vstack(rows)
    target = np.array(targets, dtype=int)
    attributes = [
        Attribute("Balance Limit", False),
        Attribute("Sex", True),
        Attribute("Education", True),
        Attribute("Marriage", True),
        Attribute("Age", False),
        Attribute("Payment 1 months late", False),
        Attribute("Payment 2 months late", False),
        Attribute("Payment 3 months late", False),
        Attribute("Payment 4 months late", False),
        Attribute("Payment 5 months late", False),
        Attribute("Payment 6 months late", False),
        Attribute("Bill 1 amount", False),
        Attribute("Bill 2 amount", False),
        Attribute("Bill 3 amount", False),
        Attribute("Bill 4 amount", False),
        Attribute("Bill 5 amount", False),
        Attribute("Bill 6 amount", False),
        Attribute("Payment 1 amount", False),
        Attribute("Payment 2 amount", False),
        Attribute("Payment 3 amount", False),
        Attribute("Payment 4 amount", False),
        Attribute("Payment 5 amount", False),
        Attribute("Payment 6 amount", False),
    ]

    return data, target, attributes

if __name__ == "__main__":
    a,b,c = load_loan()
    print("Yay!")