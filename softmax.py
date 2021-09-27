from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):

    """
    Softmax loss function, naive implementation (with loops)
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
    - gradient with respect to weights W; an array of same shape as W
    """
    loss = 0.0
    dW = np.zeros_like(W)

    # Initialize the loss and gradient to zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    
    # Compute the softmax loss and its gradient using explicit loops.           
    # Store the loss in loss and the gradient in dW. If you are not careful    
    # here, it is easy to run into numeric instability. Don't forget the        
    # regularization!
    # forward and backward
    for i in range(num_train):
        score = np.dot(X[i].T, W)
        correctScore = score[y[i]]
        loss -= np.log(np.exp(correctScore)/np.sum(np.exp(score)))
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += - X[i] + np.exp(score[j])/ np.sum(np.exp(score)) * X[i] 
            else:
                dW[:, j] += np.exp(score[j])/ np.sum(np.exp(score)) * X[i]
    # compute loss
    loss /= num_train 
    loss += reg * np.sum(W * W)
    # add reg term
    dW /= num_train
    dW += 2* reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.

    
    
    # Compute the softmax loss and its gradient using no explicit loops.  
    # Store the loss in loss and the gradient in dW. If you are not careful     
    # here, it is easy to run into numeric instability. Don't forget the        
    # regularization!
    num_train = X.shape[0]
    score = np.dot(X, W)
    correctScore = -np.sum(score[range(num_train), y])
    loss = correctScore + np.sum(np.log(np.sum(np.exp(score), axis = 1)),axis=0)
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    num_classes = W.shape[1]
    countOfX = np.zeros((num_train, num_classes))+ np.exp(score)/ np.sum(np.exp(score),axis= 1).reshape(-1,1)
    countOfX[range(num_train), y] -= 1 
    dW = np.dot(X.T, countOfX)
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW

    """
    Fully Explained at :
    https://ljvmiranda921.github.io/notebook/2017/02/14/softmax-classifier/
    """