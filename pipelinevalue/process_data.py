
import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pymongo import MongoClient

def get_data():

    # easier to work with numpy array
    data = np.genfromtxt("tagcounts.tsv", delimiter='\t', dtype=np.int32)
    ids = np.genfromtxt("tagcounts_trialids.tsv", delimiter='\n', dtype=np.str)

    X = data[::-1].astype(np.int32)

    # Y is the calculated per trial value
    Y = []
    db_stocks = MongoClient("mongodb://localhost:27017").stocks
    for id in ids:
        li = db_stocks.listed.find_one({"trials":id, "marketcapPerTrial":{"$exists":True}})
        Y.append(li["marketcapPerTrial"])
    Y = np.array(Y).astype(np.int32)

    print('X.shape before PCA')
    print(X.shape)

    # Too many tags, do dimensionality reduction just on the tags (column 4 and on ..)
    pca = PCA()
    reduced = pca.fit_transform(X)
    reduced = reduced[:, :300] # .. however much cutoff u want
    # make new X
    X = reduced
    plt.plot(pca.explained_variance_ratio_)
    plt.title('explained_variance_ratio_')
    plt.show()

    # cumulative variance
    # choose k = number of dimensions that gives us 95-99% variance
    cumulative = []
    last = 0
    for v in pca.explained_variance_ratio_:
        cumulative.append(last + v)
        last = cumulative[-1]
    plt.plot(cumulative)
    plt.title('cumulative')
    plt.show()

    print('size X: ' + str(X.shape))
    print('size Y: ' + str(Y.shape))

    return X, Y

if __name__ == '__main__':
    get_data()
