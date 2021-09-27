from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_vectorized(W, X, y, reg):

    """
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs 
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_train = X.shape[0]
    scores = np.dot(X,W)
    m = np.maximum(0, scores - scores[range(num_train), y].reshape(-1, 1) + 1.0)
    
    # Averaging over all examples
    m[range(num_train), y] = 0
    # Add regularization
    loss = np.sum(m)/num_train + reg * np.sum(W * W)

    num_class = W.shape[1]
    countOfXT = np.zeros((num_train, num_class))
    countOfXT[m>0]=1
    countOfXT[range(num_train), y]=-np.sum(countOfXT, axis = 1)
    
    dW = np.dot(X.T, countOfXT)/num_train + 2*reg*W

    return loss, dW

    '''
    FULLY EXPLAINED CODE AT:
    https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html
    '''