from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process_forlogisticregression import get_data

# get the data
Xtrain, Ytrain, Xtest, Ytest = get_data()

# randomly initialize weights
D = Xtrain.shape[1]
W = np.random.randn(D)
b = 0  # bias term

# make predictions
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)

# cross entropy
def cross_entropy(T, pY):
    return -np.mean(T * np.log(pY) + (1 - T) * np.log(1 - pY))

# train loop
train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000): # ten thousand epochs
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)

    ctrain = cross_entropy(Ytrain, pYtrain)
    ctest = cross_entropy(Ytest, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    # gradient descent
    # W -= learning_rate * (Xtrain.T.dot(pYtrain - Ytrain) - 0.1*W) # shane added regularization term
    W -= learning_rate * Xtrain.T.dot(pYtrain - Ytrain) # shane added regularization term
    b -= learning_rate * (pYtrain - Ytrain).sum()
    if i % 1000 == 0:
        print(i, ctrain, ctest)

print("Final train classification_rate:",
      classification_rate(Ytrain, np.round(pYtrain)))
print("Final test classification_rate:",
      classification_rate(Ytest, np.round(pYtest)))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()
