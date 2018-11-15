'''
Model Hypothesis: The value of a company without revenue from an approved drug is related
to the quality and number of it's clinical trials. So calculate a dollar value per trial for
each company. This assumption is more accurate as the drugs in development. Big companies with high
Market Cap (MC) are going after big targets w/ important trials. Therefore small companies with
similar trials should expect a similar large payout. To remove revenue from MC,
subtract some revenue multiplier (5 years).

DISCLAIMER: these assumptions are comletely foolish. If this is the kind of the quants on wall street do,
then they are insane.

$ / trial = MC / Ntrials

Create a neural network to learn to predict dollar value per trial:
X = (Ntrials, [tags, nlp, paper's tags]) ... each sample is a trial and the data we can gather on it.
Y = MC / N trials(company) ... the target is the dollar value per trial.

Then pass a company's pipeline forward through this network to predict a learned pipeline value.
Sort stocks by pipeline value for investment decisions.
'''

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Input, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D
from keras import metrics
from keras import optimizers
from process_data import get_data
import matplotlib.pyplot as plt
import json
from sklearn.utils import shuffle
import operator
from tabulate import tabulate

from pymongo import MongoClient
db = MongoClient("mongodb://localhost:27017").stocks
# TODO: still need a file based version of the data for cloud gpu

###############################

# configuration / hyperparameters
TRAINING_SPLIT = 0.80 # raise to 1 when final model train on all data
BATCH_SIZE = 128
EPOCHS = 400
LEARNING_RATE = 0.0001
OPTIMIZER = optimizers.RMSprop(lr=LEARNING_RATE, rho=0.9, epsilon=None, decay=0.0)
HIDDEN_LAYERS = 8
HIDDEN_UNITS = 64
DROPOUT = 0.5
ACTIVATION = 'relu'
LOSS_FUNCTION = 'mean_squared_error'

# Principle components analysis of tag data if True
PCAtags=False
PCAvalue = 100

# get the data
X, Y, Ymean, Ystd, ids_today, mgs_to_trialid, Xtoday = get_data(PCAtags=PCAtags, PCAvalue=PCAvalue)
X, Y = shuffle(X, Y) # shuffle but keep indexes together
Ntrain = int(TRAINING_SPLIT * len(X)) # give it all the data to train
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

# get shapes
N, D = X.shape
print('X.shape N,D:', X.shape)

# the model will be a sequence of layers
model = Sequential()
# input layer
model.add(Dense(units=HIDDEN_UNITS, input_dim=D, activation=ACTIVATION))
for _ in range(HIDDEN_LAYERS):
    model.add(Dense(units=HIDDEN_UNITS, activation=ACTIVATION))
model.add(Dropout(DROPOUT))
# no activation on output layer for regression
model.add(Dense(1))

print(model.summary())

# Compile model
model.compile(
    loss=LOSS_FUNCTION,
    optimizer=OPTIMIZER,
    metrics=[metrics.mae]
)

r = model.fit(
    Xtrain,
    Ytrain,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(Xtest, Ytest)
)

# predict from today's trials only
print('calculating/predicting for today ...')
ynew = model.predict(Xtoday)

# build the pipeline values for each company based on Today's data only, not historical
mgPipeline = {}
for mgname in mgs_to_trialid:
    mgPipeline[mgname] = 0
    trials = mgs_to_trialid[mgname]
    for t in trials:
        for num, id in enumerate(ids_today):
            if (t == id):
                Z = ynew[num][0]
                mc = ((Z-1) * Ystd) + Ymean # remember to unshift the mean
                mgPipeline[mgname] = mgPipeline[mgname] + int(mc)


# calculate the percent diffs, using today's data in listed, not historical
mgDiffs = {}
for mg in mgPipeline:
    li = db.listed.find_one({"medicalgroups":mg})
    mcReal = li['marketcap']
    diff = ( (mgPipeline[mg] - mcReal) / (mcReal+1) )
    mgDiffs[mg] = diff

sorted_x = sorted(mgDiffs.items(), key=operator.itemgetter(1), reverse=False)
tot = []
for i in sorted_x:
    tot.append(i)

df = pd.DataFrame(tot, columns=["Name", "Mult"])
print(tabulate(df, headers='keys', tablefmt='psql'))

# print the difference in change
changeabsolute = ((r.history['val_mean_absolute_error'][0] - r.history['val_mean_absolute_error'][-1]) / r.history['val_mean_absolute_error'][0]) * 100
changeabsolute2 = ((r.history['mean_absolute_error'][0] - r.history['mean_absolute_error'][-1]) / r.history['mean_absolute_error'][0]) * 100
print('absolute error difference from start (validation): ', "%.2f" % changeabsolute, '%')
print('absolute error difference from start (train): ', "%.2f" % changeabsolute2, '%')

# print(r.history.keys())
plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.plot(r.history['mean_absolute_error'])
plt.plot(r.history['val_mean_absolute_error'])
plt.title('model loss/accuracy (absolute error)')
plt.xlabel('epoch')
plt.legend(['train loss', 'test loss', 'train err', 'test err'], loc='upper left')
plt.show()

# Notes
# save and load keras models
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# hyperparam op: https://github.com/autonomio/talos
# https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53
# Collection of keras model examples
# https://gist.github.com/candlewill/552fa102352ccce42fd829ae26277d24
