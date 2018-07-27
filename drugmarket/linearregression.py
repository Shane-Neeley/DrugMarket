# -*- coding: utf-8 -*-
# from core import runCore
# df = runCore()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# https://medium.com/we-are-orb/multivariate-linear-regression-in-python-without-scikit-learn-7091b1d45905


df = pd.read_csv("drugmarket_dataframe.tsv", sep='\t')

# limit high level outliers
df = df[df['MC'] > 0]
df = df[ (df['Phase 4'] > 1) | (df['Phase 3'] > 1) | (df['Phase 2'] > 1) | (df['Phase 1'] > 1)] # has any trials
df = df[df['Symbol'] != "GILD"]
df = df[df['Symbol'] != "SYK"]
df = df[df['Symbol'] != "MDT"]


print(df)
df = df.sample(frac=1) # this is a row shuffle, mixes up data, because it's sorted by marketcap now
print(df)

# Note this is clearly not a linear relationship!
# I think i need to do PCA, w/ additional data on the stock to find out how much influence trials has on marketcap

def normalize(a):
    return (a - a.mean()) / a.std()

X1 = df['Phase 1'].values
X2 = df['Phase 2'].values
X3 = df['Phase 3'].values
X4 = df['Phase 4'].values
X1 = normalize(X1)
X2 = normalize(X2)
X3 = normalize(X3)
X4 = normalize(X4)

ones = np.ones((X1.shape[0]))
X = np.array([ones,X1,X2,X3,X4])
#we need to normalize the features using mean normalization


N = X.shape[1]
D = X.shape[0]

Y = df['MC'].values

plt.scatter(X1, Y, color='red')
plt.scatter(X2, Y, color='blue')
plt.scatter(X3, Y, color='green')
plt.scatter(X4, Y, color='black')
plt.title("Data")
plt.show()
# Note this is clearly not a linear relationship!

#set hyper parameters
learning_rate = 0.001
iters = 1000

# let's try gradient descent
costs = []  # keep track of squared error cost
w = np.random.randn(D) / np.sqrt(D)  # randomly initialize w
# Take steps
for t in range(iters):
    # update w
    Yhat = X.T.dot(w)
    delta = Yhat - Y
    w = w - learning_rate * X.dot(delta)
    # find and store the cost
    mse = delta.dot(delta) / N
    costs.append(mse)

# plot the costs
plt.plot(costs)
plt.title("Costs (MSE)")
plt.show()
print("final w:", w)
# final w: [ 8.13113518e+09  3.04840577e+10 -1.60066020e+10 -5.94552095e+09 4.35092712e+08] bias,p1,p2,p3,p4 .. p4 is a big factor.
# plot prediction vs target
plt.plot(Yhat, label='prediction')
plt.plot(Y, label='target')
plt.legend()
plt.show()
