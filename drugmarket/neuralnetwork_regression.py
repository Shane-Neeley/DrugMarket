from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from sklearn.utils import shuffle
from process_data import get_data

Xtrain, Ytrain, Xtest, Ytest, datatrain, datatest = get_data(regression=True)

X = Xtrain
Y = Ytrain
# normalize, keep original to unscale later
Yorig = Y
Y = ( Y-np.min(Y)) / (np.max(Y)-np.min(Y) )

D = X.shape[1]
K = len(set(Ytrain) | set(Ytest))
M = 10  # num hidden units
# layer 1
W = np.random.randn(D, M) / np.sqrt(D)
b = np.zeros(M)
# layer 2
V = np.random.randn(M) / np.sqrt(M)
c = 0

def forward(X):
    Z = X.dot(W) + b
    Z = Z * (Z > 0)  # relu
    # Z = np.tanh(Z)
    Yhat = Z.dot(V) + c
    return Z, Yhat

# how to train the params
def derivative_V(Z, Y, Yhat):
    return (Y - Yhat).dot(Z)

def derivative_c(Y, Yhat):
    return (Y - Yhat).sum()

def derivative_W(X, Z, Y, Yhat, V):
    # dZ = np.outer(Y - Yhat, V) * (1 - Z * Z) # this is for tanh activation
    dZ = np.outer(Y - Yhat, V) * (Z > 0)  # relu
    return X.T.dot(dZ)

def derivative_b(Z, Y, Yhat, V):
    # dZ = np.outer(Y - Yhat, V) * (1 - Z * Z) # this is for tanh activation
    dZ = np.outer(Y - Yhat, V) * (Z > 0)  # this is for relu activation
    return dZ.sum(axis=0)

l1 = 0.0
learning_rate=1e-4

def update(X, Z, Y, Yhat, W, b, V, c):
    gV = derivative_V(Z, Y, Yhat)
    gc = derivative_c(Y, Yhat)
    gW = derivative_W(X, Z, Y, Yhat, V)
    gb = derivative_b(Z, Y, Yhat, V)

    V += learning_rate * (gV + l1*np.sign(V))
    c += learning_rate * gc
    W += learning_rate * (gW + l1*np.sign(W))
    b += learning_rate * gb

    return W, b, V, c

# so we can plot the costs later
def get_cost(Y, Yhat):
    return ((Y - Yhat)**2).mean()

# run a training loop
# plot the costs
# and plot the final result
costs = []
testcosts = []
for i in range(2000):
    Z, Yhat = forward(X)
    W, b, V, c = update(X, Z, Y, Yhat, W, b, V, c)
    cost = get_cost(Y, Yhat)
    costs.append(cost)

    # Ztest, Yhat_test = forward(Xtest)
    # testcost = get_cost(Ytest, Yhat_test)
    # testcosts.append(testcost)
    if i % 250 == 0:
        print(cost)

# print unnormalized data
pred = Yhat*Yorig
diff = pred-Yorig
shanes_arbitrary_scalar = 100
diffmultiple = (pred / Yorig) * shanes_arbitrary_scalar
pred = pred.astype(np.int64)
totaltrials = np.sum(datatrain[:,3:7], axis=1)
result = np.column_stack((datatrain[:,1], datatrain[:,2], Yorig, pred, diffmultiple, totaltrials))
df = pd.DataFrame(result, columns=["Company", "Stock", "MarketCap", "Prediction", "Diff X", "Total Trials"])
# Sorting by those who the neural network predicted to have a higher value
df = df.sort_values(['Diff X'], ascending=[False])

# look at just sub billion valuation companies
df2 = df[df["MarketCap"] < 1e9]
# print w/ commas dollar format
df2['MarketCap'] = df2.apply(lambda x: "{:,}".format(x['MarketCap']), axis=1)
df2['Prediction'] = df2.apply(lambda x: "{:,}".format(x['Prediction']), axis=1)

print(tabulate(df2, headers='keys', tablefmt='psql'))

# plot the costs
# plt.plot(costs)
# plt.show()
