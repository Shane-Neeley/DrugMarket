from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
import os

# normalize numerical columns
# one-hot categorical columns

def get_data():
    df = pd.read_csv('drugmarket_dataframe.tsv', dtype={'MC':np.int32}, sep="\t")

    # easier to work with numpy array
    data = df.values

    # create a final output column of a category
    # 1 = >$1Billion market cap, 0 = less
    categ = np.array(data[:, -1] > 1e9, dtype=bool).astype(int)
    categ = np.array([categ]).T
    data = np.concatenate((data,categ),1)

    # shuffle it
    np.random.shuffle(data)

    # split features and labels
    X = data[:, 3:-2].astype(np.int32) # this just pulled excluded the last two columns
    print(X.shape)

    Y = data[:, -1].astype(np.int32) # this is the last column

    # one-hot encode the categorical data
    # create a new matrix X2 with the correct number of columns
    # N, D = X.shape
    # X2 = np.zeros((N, D + 3))
    # X2[:, 0:(D - 1)] = X[:, 0:(D - 1)]  # non-categorical
    #
    # # one-hot
    # for n in range(N):
    #     t = int(X[n, D - 1])
    #     X2[n, t + D - 1] = 1


    # assign X2 back to X, since we don't need original anymore
    # X = X2

    # split train and test
    # has total 354 rows
    Xtrain = X[:-100]
    Ytrain = Y[:-100]
    Xtest = X[-100:]
    Ytest = Y[-100:]

    # normalize phase columns
    for i in (0, 1, 2, 3):
        m = Xtrain[:, i].mean()
        s = Xtrain[:, i].std()
        Xtrain[:, i] = (Xtrain[:, i] - m) / s
        Xtest[:, i] = (Xtest[:, i] - m) / s

    return Xtrain, Ytrain, Xtest, Ytest
