
import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pymongo import MongoClient
from datetime import date

def get_data(PCAtags = True, PCAvalue = 300):
    # get X, Y as the total of all acquired dates
    X = []
    Y = []
    # put today's data in it's own arrays for prediction
    today = date.today().strftime('%m-%d-%Y')
    mgs_to_trialid = {}
    Xtoday = []
    ids_today = []

    db = MongoClient("mongodb://localhost:27017").stocks
    for coll in db.collection_names():
        if 'tagdata-' in coll:
            for d in db[coll].find({}):
                X.append(d['data'])
                Y.append(d['marketcapPerTrial'])
                # collect into today's as well (also train on today from above?)
                if coll.split('tagdata-')[1] == today:
                    Xtoday.append(d['data'])
                    ids_today.append(d['id'])
                    if d['medicalgroup'] not in mgs_to_trialid:
                        mgs_to_trialid[d['medicalgroup']] = [d['id']]
                    else:
                        mgs_to_trialid[d['medicalgroup']].append(d['id'])

    # make'em numpy
    ids_today = np.array(ids_today, dtype=np.str)
    Xtoday = np.array(Xtoday, dtype=np.int32)

    X = np.array(X, dtype=np.int32)
    Y = np.array(Y, dtype=np.int32)
    # normalize Y
    Ystd = Y.std()
    Ymean = Y.mean()
    Y = ( (Y - Ymean) / Ystd ) + 1

    if PCAtags:
        print('X.shape before PCA')
        print(X.shape)
        # columns 0,4 are phase
        Xtags = X[:,5:]
        Xphase = X[:,:4]
        Xtodaytags = Xtoday[:,5:]
        Xtodayphase = Xtoday[:,:4]

        # Too many tags, do dimensionality reduction just on the tags (column 4 and on ..)
        pca = PCA()
        reduced = pca.fit_transform(Xtags)
        reduced = reduced[:, :PCAvalue] # .. however much cutoff u want
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

        pca = PCA()
        reduced = pca.fit_transform(Xtodaytags)
        reduced = reduced[:, :PCAvalue]
        Xtoday = np.concatenate((Xtodayphase, reduced), 1)

    print('size X: ' + str(X.shape))
    print('size Y: ' + str(Y.shape))
    print('size Xtoday: ' + str(Xtoday.shape))

    return X, Y, Ymean, Ystd, ids_today, mgs_to_trialid, Xtoday

if __name__ == '__main__':
    get_data()
