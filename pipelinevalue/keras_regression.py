import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import metrics
from process_data import get_data
import matplotlib.pyplot as plt
import json
from pymongo import MongoClient
from sklearn.utils import shuffle
import operator
from tabulate import tabulate
client = MongoClient("mongodb://localhost:27017")
db = client.stocks


X, Y, Ymean, Ystd, ids = get_data()
X, Y, ids = shuffle(X, Y, ids) # shuffle but keep indexes together
Ntrain = int(0.75 * len(X)) # give it all the data to train
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
# idsTrain = ids[:Ntrain]
# idsTest = ids[Ntrain:]

# get shapes
N, D = X.shape

# the model will be a sequence of layers
model = Sequential()
#layer 1
model.add(Dense(units=32, input_dim=D, activation = 'relu'))
model.add(Dropout(0.5))
#layer 2
model.add(Dense(units=32, activation = 'relu'))
#layer 3
model.add(Dense(units=32, activation = 'relu'))
model.add(Dropout(0.5))
#layer 4
model.add(Dense(units=32, activation = 'relu'))
model.add(Dropout(0.5))
# no activation on final layer for regression
model.add(Dense(1))


# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer='rmsprop',
    # metrics=[metrics.mae]
)

r = model.fit(
    Xtrain,
    Ytrain,
    epochs=200,
    batch_size=500,
    validation_data=(Xtest, Ytest)
)

print('calculating/predicting ...')

# predict from all trials
ynew = model.predict(X)

# build the pipeline values for each company
mgs = np.genfromtxt("mgs_to_trialid.tsv", delimiter='\n', dtype=np.str)
mgPipeline = {}
for mg in mgs:
    mg = mg.split('\t')
    mgname = mg[0]
    mgPipeline[mgname] = 0
    trials = mg[1:]
    for t in trials:
        for num, id in enumerate(ids):
            if (t == id):
                Z = ynew[num][0]
                mc = ( (Z - 1) * Ystd) + Ymean
                mgPipeline[mgname] = mgPipeline[mgname] + int(mc)


# calculate the percent diffs
mgDiffs = {}
for mg in mgPipeline:
    li = db.listed.find_one({"medicalgroups":mg})
    mcReal = li['marketcap']
    diff = ( (mgPipeline[mg] - mcReal) / (mcReal+1) )
    mgDiffs[mg] = diff

sorted_x = sorted(mgDiffs.items(), key=operator.itemgetter(1), reverse=False)
tot = []
for i in sorted_x:
    tot.append([i[0], "{:,}".format(int(i[1])) + 'X'])
df = pd.DataFrame(tot, columns=["Name", "Mult"])
print(tabulate(df, headers='keys', tablefmt='psql'))


plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



# save and load keras models
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
