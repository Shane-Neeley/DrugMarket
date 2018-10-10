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
from keras import metrics
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

X, Y, Ymean, Ystd, ids_today, mgs_to_trialid, Xtoday = get_data(PCAtags=True, PCAvalue=300)
X, Y = shuffle(X, Y) # shuffle but keep indexes together
Ntrain = int(0.80 * len(X)) # give it all the data to train
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
# idsTrain = ids[:Ntrain]
# idsTest = ids[Ntrain:]

# get shapes
N, D = X.shape

# the model will be a sequence of layers
model = Sequential()
# input layer
model.add(Dense(units=64, input_dim=D, activation='relu'))
hidden_layers = 7
for _ in range(hidden_layers):
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.2))
# no activation on output layer for regression
model.add(Dense(1))

# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer='rmsprop',
    metrics=[metrics.mae]
)

r = model.fit(
    Xtrain,
    Ytrain,
    epochs=20,
    batch_size=128,
    validation_data=(Xtest, Ytest)
)

# predict from today's trials only
print('calculating/predicting ...')
ynew = model.predict(Xtoday)
print(ynew)

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

plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Notes
# save and load keras models
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# hyperparam op: https://github.com/autonomio/talos
# https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53
# example of model params for grid search
# p = {
#      'lr': (0.8, 1.2, 3),
#      'first_neuron':[4, 8, 16, 32, 64],
#      'hidden_layers':[0, 1, 2],
#      'batch_size': (1, 5, 5),
#      'epochs': [50, 100, 150],
#      'dropout': (0, 0.2, 3),
#      'weight_regulizer':[None],
#      'emb_output_dims': [None],
#      'shape':['brick','long_funnel'],
#      'kernel_initializer': ['uniform','normal'],
#      'optimizer': [Adam, Nadam, RMSprop],
#      'losses': [binary_crossentropy],
#      'activation':[relu, elu],
#      'last_activation': [sigmoid]
# }
# Collection of keras model examples
# https://gist.github.com/candlewill/552fa102352ccce42fd829ae26277d24
