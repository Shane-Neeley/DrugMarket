# For the class Data Science: Practical Deep Learning Concepts in Theano and TensorFlow
# https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
# https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from process_data import get_data
from tf_ann import ANN # internal module

def random_search():

    X, Y, data = get_data()
    X, Y = shuffle(X, Y)
    Ntrain = int(0.75 * len(X))
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    # Make copies of the small data (because variance matters?)
    Xtrain = np.concatenate((Xtrain,Xtrain,Xtrain), 0)
    Ytrain = np.concatenate((Ytrain,Ytrain,Ytrain), 0)

    print('size Xtrain: ' + str(Xtrain.shape))
    print('size Ytrain: ' + str(Ytrain.shape))
    print('size Xtest: ' + str(Xtest.shape))
    print('size Ytest: ' + str(Ytest.shape))

    # starting hyperparameters
    M = 20 # hidden units
    nHidden = 2 # hidden layers
    log_lr = -4 # learning rate
    log_l2 = -2  # l2 regularization, since we always want it to be positive
    max_tries = 30

    # loop through all possible hyperparameter settings
    best_validation_rate = 0
    best_hls = None
    best_lr = None
    best_l2 = None
    validation_accuracies = []
    for _ in range(max_tries):
        print('on try: ' + str(_+1) + '/' + str(max_tries))
        model = ANN([M] * nHidden)
        # choose params randomly on log base 10 scale
        model.fit(
            Xtrain, Ytrain,
            learning_rate=10**log_lr, reg=10**log_l2,
            mu=0.99, epochs=4000, show_fig=True
        )
        validation_accuracy = model.score(Xtest, Ytest)
        train_accuracy = model.score(Xtrain, Ytrain)
        print(
            "validation_accuracy: %.3f, train_accuracy: %.3f, settings: %s (layers), %s (log_lr), %s (log_l2)" %
            (validation_accuracy, train_accuracy,
             [M] * nHidden, log_lr, log_l2)
        )
        # keep track of all
        validation_accuracies.append(validation_accuracy)

        # keep the best parameters, then make modifications to them
        if validation_accuracy > best_validation_rate:
            best_validation_rate = validation_accuracy
            best_M = M
            best_nHidden = nHidden
            best_lr = log_lr
            best_l2 = log_l2

        # select new hyperparams
        nHidden = best_nHidden + np.random.randint(-1, 2)  # -1, 0, or 1, add, remove or keep same the layers
        nHidden = max(1, nHidden)
        M = best_M + np.random.randint(-1, 2) * 10
        M = max(10, M)
        log_lr = best_lr + np.random.randint(-1, 2)
        log_l2 = best_l2 + np.random.randint(-1, 2)


    # TODO: save these in mongodb, then read them and see if we beat it, in a new file run forward on best params
    print("Best validation_accuracy:", best_validation_rate)
    print("Mean validation_accuracy:", np.mean(validation_accuracies))
    print("Best settings:")
    print("Best M (hidden units):", best_M)
    print("Best nHidden (hidden layers):", best_nHidden)
    print("Best learning_rate:", best_lr)
    print("Best l2 regularization:", best_l2)


if __name__ == '__main__':
    random_search()
