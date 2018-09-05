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
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

def random_search():

    X, Y, data = get_data()
    X, Y = shuffle(X, Y)
    Ntrain = int(0.75 * len(X))
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    # Make copies of the small data (because variance matters?)
    # Xtrain = np.concatenate((Xtrain,Xtrain,Xtrain), 0)
    # Ytrain = np.concatenate((Ytrain,Ytrain,Ytrain), 0)

    N = Xtrain.shape[0]
    D = Xtrain.shape[1]
    K = len(set(Ytrain)) # classes, only 0,1 here so len=2

    Ntest = Xtrain.shape[0]
    Dtest = Xtrain.shape[1]

    print('size Xtrain: ' + str(Xtrain.shape))
    print('size Ytrain: ' + str(Ytrain.shape))
    print('size Xtest: ' + str(Xtest.shape))
    print('size Ytest: ' + str(Ytest.shape))

    # turn Y into an indicator matrix for training
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Ytrain[i]] = 1

    Ttest = np.zeros((Ntest, K))
    for i in range(Dtest):
        Ttest[i, Ytest[i]] = 1


    # tensor flow variables are not the same as regular Python variables
    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def forward(X, W1, b1, W2, b2):
        Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
        return tf.matmul(Z, W2) + b2

    # starting hyperparameters
    M = 20 # hidden units
    nHidden = 1 # hidden layers
    log_lr = -4 # learning rate
    log_l2 = -2  # l2 regularization, since we always want it to be positive
    max_tries = 30

    # loop through all possible hyperparameter settings
    best_validation_rate = 0
    # best_nHidden = None
    best_M = None
    best_lr = None
    best_l2 = None
    validation_accuracies = []

    for _ in range(max_tries):
        print('on try: ' + str(_+1) + '/' + str(max_tries))

        tfX = tf.placeholder(tf.float32, [None, D])
        tfY = tf.placeholder(tf.float32, [None, K])
        tfXtest = tf.placeholder(tf.float32, [None, D])
        tfYtest = tf.placeholder(tf.float32, [None, K])

        W1 = init_weights([D, M])  # create symbolic variables
        b1 = init_weights([M])
        W2 = init_weights([M, K])
        b2 = init_weights([K])

        logits = forward(tfX, W1, b1, W2, b2)

        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tfY,
                logits=logits
            )
        )

        # train_op = tf.train.GradientDescentOptimizer(10**log_lr).minimize(cost)  # construct an optimizer
        # input parameter is the learning rate
        train_op = tf.train.RMSPropOptimizer(10**log_lr, decay=0.99, momentum=0.9).minimize(cost)

        predict_op = tf.argmax(logits, 1)
        # input parameter is the axis on which to choose the max

        # just stuff that has to be done
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        epochs = 3000
        for i in range(epochs):
            sess.run(train_op, feed_dict={tfX: Xtrain, tfY: T})
            pred = sess.run(predict_op, feed_dict={tfX: Xtrain, tfY: T})
            test = sess.run(predict_op, feed_dict={tfX: Xtest, tfY: Ttest})

        train_accuracy = np.mean(Ytrain == pred)
        validation_accuracy = np.mean(Ytest == test)
        print("Accuracy:", validation_accuracy)
        print(
            "validation_accuracy: %.3f, train_accuracy: %.3f, settings: %s (layers), %s (log_lr), %s (log_l2)" %
            (validation_accuracy, train_accuracy,
             [M] * nHidden, log_lr, log_l2)
        )

        # keep the best parameters, then make modifications to them
        if validation_accuracy > best_validation_rate:
            best_validation_rate = validation_accuracy
            best_M = M
            # best_nHidden = nHidden
            best_lr = log_lr
            best_l2 = log_l2

        # select new hyperparams
        # nHidden = best_nHidden + np.random.randint(-1, 2)  # -1, 0, or 1, add, remove or keep same the layers
        # nHidden = max(1, nHidden)
        M = best_M + np.random.randint(-1, 2) * 10
        M = max(10, M)
        log_lr = best_lr + np.random.randint(-1, 2)
        log_l2 = best_l2 + np.random.randint(-1, 2)


    # TODO: save these in mongodb, then read them and see if we beat it, in a new file run forward on best params
    print("Best validation_accuracy:", best_validation_rate)
    # print("Mean validation_accuracy:", np.mean(validation_accuracies))
    print("Best settings:")
    print("Best M (hidden units):", best_M)
    # print("Best nHidden (hidden layers):", best_nHidden)
    print("Best learning_rate:", best_lr)
    print("Best l2 regularization:", best_l2)


if __name__ == '__main__':
    random_search()
