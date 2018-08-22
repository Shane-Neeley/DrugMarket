from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process_forlogisticregression import get_data

def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

Xtrain, Ytrain, Xtest, Ytest, datatrain, datatest = get_data()
D = Xtrain.shape[1]
K = len(set(Ytrain) | set(Ytest))
M = 100  # num hidden units

# convert to indicator
Ytrain_ind = y2indicator(Ytrain, K)
Ytest_ind = y2indicator(Ytest, K)

# randomly initialize weights
W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

# make predictions
def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    # return softmax(Z.dot(W2) + b2), Z
    return sigmoid(Z.dot(W2) + b2), Z # doing binary classification so can use sigmoid

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)

# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(T, pY):
    return -np.mean(T * np.log(pY))

# train loop
train_costs = []
test_costs = []
learning_rate = 0.001
epochs = 2000
l1 = 0  # try different values l1 regularization - what effect does it have on w?
for i in range(epochs):
    pYtrain, Ztrain = forward(Xtrain, W1, b1, W2, b2)
    pYtest, Ztest = forward(Xtest, W1, b1, W2, b2)

    ctrain = cross_entropy(Ytrain_ind, pYtrain)
    ctest = cross_entropy(Ytest_ind, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    # gradient descent
    W2 -= learning_rate * (Ztrain.T.dot(pYtrain - Ytrain_ind) + l1 * np.sign(W2))
    b2 -= learning_rate * (pYtrain - Ytrain_ind).sum(axis=0)
    dZ = (pYtrain - Ytrain_ind).dot(W2.T) * (1 - Ztrain * Ztrain)
    W1 -= learning_rate * (Xtrain.T.dot(dZ) + l1 * np.sign(W1))
    b1 -= learning_rate * dZ.sum(axis=0)
    if i % 200 == 0:
        print(i, ctrain, ctest)

print("Final train classification_rate:",
      classification_rate(Ytrain, predict(pYtrain)))
print("Final test classification_rate:",
      classification_rate(Ytest, predict(pYtest)))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()
