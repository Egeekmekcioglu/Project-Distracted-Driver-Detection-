from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
from linear_svm import *
from softmax import *
from past.builtins import xrange


class LinearClassifier(object):

    def __init__(self):
        self.W = None      # Initialize W.
        

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):

        
        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # Assume y takes values, where K is number of classes.
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)
            loss_history = []       # Stochastic Grad. Des. to optimize W.
        for it in range(num_iters):
            X_batch = None
            y_batch = None
            
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            

            index = np.random.choice(num_train, batch_size,replace = True)
            X_batch = X[index]
            y_batch = y[index]

            # EVALUATE LOSS AND GRADIENT
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # Update the weights using the gradient and the learning rate.
            self.W -= learning_rate * grad
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):

        
        '''
        USE THE TRAINED WEIGHTS OF THIS LINEAR CLASSIFIER TO PREDICT LABELS FOR DATA POINTS.
        INPUTS:
        -X: D x N ARRAY OF TRAINING DATA.
        RETURNs:
        -y_pred : PREDİCTED LABELS FOR THE DATA İN X.y_pred İS A 1-DIMENSIONAL ARRAY OF LENGTH N.
        
        '''
        
        y_pred = np.zeros(X.shape[0])

         # Implement this method. Store the predicted labels in y_pred. 
        y_pred = np.argmax(np.dot(X,self.W), axis = 1)

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.
        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.
        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):
    #A subclass that uses the Multiclass SVM loss function 
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    #A subclass that uses the Softmax + Cross-entropy loss function   
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)