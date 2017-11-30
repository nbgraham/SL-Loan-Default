import numpy as np
from sklearn.model_selection import train_test_split
from test_nn import Network

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return z*(1-z)

with open('UCI_Credit_Card.csv','r') as f:
    data = f.read().split('\n')
    data = np.array(data)
    data = data[1:]


formatted_data = [[float(i) for i in np.array(d.split(","))] for d in data]

train,test = train_test_split(formatted_data,test_size=0.2)

train_targets = np.array([[element[-1]] for element in train])
train_features = np.array([element[1:-1] for element in train])

test_targets = np.array([[element[-1]] for element in test])
test_features = np.array([element[1:-1] for element in test])

np.random.seed(35)

# network layers
n_input = 23
n_hidden = 30
n_output = 1

# learning rate
alpha = 0.001

n_epochs = 100

# weights
w0 = 2 * np.random.random((n_input,n_hidden)) - 1
w1 = 2 * np.random.random((n_hidden,n_output)) - 1

# training
for i in range(n_epochs):
    err = 0
    w0_sum = 0
    w1_sum = 0
    for j in range(len(train_features)):
        # layers
        l0 = train_features[j].reshape((-1,23))
        l1 = sigmoid(np.dot(l0,w0))
        l2 = sigmoid(np.dot(l1,w1))

        # backprop
        l2_error = train_targets[j] - l2
        l2_delta = l2_error * sigmoid_prime(l2)
        l1_error = l2_delta.dot(w1.T)
        l1_delta = l1_error * sigmoid_prime(l1)

        w1_sum += alpha*(l1.T.dot(l2_delta))
        w0_sum += alpha*(l0.T.dot(l1_delta))

        err += np.mean(np.abs(l2_error))

    # update weights
    w0 += w0_sum
    w1 += w1_sum

    if i % (n_epochs/10) == 0:
        print("error: ", err/len(train_features))

print(np.sum(l2,axis=0))
