from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
import os
from tabulate import tabulate

# normalize numerical columns
# one-hot categorical columns

def get_data(classification=True, regression=False):
    df = pd.read_csv('drugmarket_dataframe.tsv', dtype={'MC':np.int64}, sep="\t")

    # remove outliers
    df = df[df['MC'] > 0]
    df = df[ (df['Phase 4'] > 0) | (df['Phase 3'] > 0) | (df['Phase 2'] > 0) | (df['Phase 1'] > 0)] # has any trials
    # df = df[ (df['Phase 4'] < 500) | (df['Phase 3'] < 500) | (df['Phase 2'] < 500) | (df['Phase 1'] < 500)] # has too many trials
    df = df[df['Symbol'] != "SYK"] # stryker an outlier

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
    X = data[:, 3:-2].astype(np.int64) # this just pulled excluded the last two columns

    if (classification == True):
        Y = data[:, -1].astype(np.int64) # this is the last column, 0 or 1 class for billion dollar valuation
    if (regression == True):
        Y = data[:, -2].astype(np.int64) # continuous value for marketcap

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

    # split train and test, then convert to floats for normalization
    # has total 363 rows, same some for test
    limit = -50
    Xtrain = X[:limit].astype(np.float32)
    Ytrain = Y[:limit]
    datatrain = data[:limit]
    Xtest = X[limit:].astype(np.float32)
    Ytest = Y[limit:]
    datatest = data[limit:]

    print('size Xtrain: ' + str(Xtrain.shape))
    print('size Ytrain: ' + str(Ytrain.shape))
    print('size Xtest: ' + str(Xtest.shape))
    print('size Ytest: ' + str(Ytest.shape))

    # normalize phase columns by X - mean / std
    for i in (0, 1, 2, 3):
        m = Xtrain[:, i].mean()
        s = Xtrain[:, i].std()
        Xtrain[:, i] = (Xtrain[:, i] - m) / s
        Xtest[:, i] = (Xtest[:, i] - m) / s

    return Xtrain, Ytrain, Xtest, Ytest, datatrain, datatest
