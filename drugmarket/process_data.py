from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
import os
from tabulate import tabulate
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# normalize numerical columns
# one-hot categorical columns

def get_data(classification=True, regression=False):
    df = pd.read_csv('drugmarket_dataframe.tsv', dtype={'MC':np.int64}, sep="\t")

    # remove outliers
    df = df[df['MC'] > 0]
    # df = df[ (df['Phase 4'] > 0) | (df['Phase 3'] > 0) | (df['Phase 2'] > 0) | (df['Phase 1'] > 0)] # has any trials
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

    # print(df)
    print(X)
    # print('X.shape before')
    # print(X.shape)

    # Too many tags, do dimensionality reduction just on the tags (column 4 and on ..)
    pca = PCA()
    reduced = pca.fit_transform(X[:, 4:])
    # print('reduced.shape before')
    # print(reduced.shape)
    # plt.scatter(reduced[:,0], reduced[:,1], s=100, c=Y, alpha=0.5)
    # plt.title('reduced')
    # plt.show()
    reduced = reduced[:, :25] # .. however much cutoff u want
    # print('reduced.shape after cutoff')
    # print(reduced.shape)
    X = np.concatenate((X[:,:4], reduced),1)
    # print('X.shape after concatenate')
    # print(X.shape)
    # print(X)
    # plt.plot(pca.explained_variance_ratio_)
    # plt.title('explained_variance_ratio_')
    # plt.show()
    # cumulative variance
    # choose k = number of dimensions that gives us 95-99% variance
    cumulative = []
    last = 0
    for v in pca.explained_variance_ratio_:
        cumulative.append(last + v)
        last = cumulative[-1]
    # plt.plot(cumulative)
    # plt.title('cumulative')
    # plt.show()


    print('size X: ' + str(X.shape))
    print('size Y: ' + str(Y.shape))

    # normalize phase columns by X - mean / std
    for i in (0, 1, 2, 3):
        m = X[:, i].mean()
        s = X[:, i].std()
        X[:, i] = (X[:, i] - m) / s

    return X, Y, data

if __name__ == '__main__':
    get_data()
