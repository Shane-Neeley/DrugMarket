
import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pymongo import MongoClient

def get_data(PCAtags = True):
    ids = []
    # get X, Y as the total of all acquired dates
    X = []
    Y = []
    db = MongoClient("mongodb://localhost:27017").stocks
    for coll in db.collection_names():
        if 'tagdata-' in coll:
            for d in db[coll].find({"id":{"$ne":"headers"}}):
                ids.append(d['id'])
                X.append(d['data'])
                Y.append(d['marketcapPerTrial'])

    ids = np.array(ids, dtype=np.str)
    X = np.array(X, dtype=np.int32)
    Y = np.array(Y, dtype=np.int32)

    Ystd = Y.std()
    Ymean = Y.mean()
    Y = ( (Y - Ymean) / Ystd ) + 1

    print('X.shape before PCA')
    print(X.shape)

    if PCAtags:
        # columns 0,4 are phase
        Xtags = X[:,5:]
        Xphase = X[:,:4]

        # Too many tags, do dimensionality reduction just on the tags (column 4 and on ..)
        pca = PCA()
        reduced = pca.fit_transform(Xtags)
        reduced = reduced[:, :300] # .. however much cutoff u want
        # make new X
        X = np.concatenate((Xphase, reduced), 1)
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

    return X, Y, Ymean, Ystd, ids

if __name__ == '__main__':
    get_data()
