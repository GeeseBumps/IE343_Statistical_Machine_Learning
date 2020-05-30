import numpy as np
import pandas as pd
import math

class LogisticRegressor():
    def __init__(self,w=None):
        self.w = w

    def fit(self, X, y,lr,epoch_num):

        #training #TODO
        w = np.zeros((np.size(X, 1),3))
        print(w)
        one_hot_y=np.zeros((np.size(X,0),3))
        for i in range(len(y)):
            if y[i]==0:
                one_hot_y[i][0] +=1
            elif y[i]==1:
                one_hot_y[i][1] += 1
            else:
                one_hot_y[i][2] += 1
        for i in range(epoch_num):
            w_prev = np.copy(w)
            input = X.dot(w)
            softmax = (np.exp(input.T)/np.sum(np.exp(input),axis=1)).T
            grad = X.T.dot((softmax - one_hot_y))
            w -= lr * grad

            if np.allclose(w, w_prev):
                break

        self.w=w


    def predict(self, X):
       #prediction #TODO
        input=X.dot(self.w)
        predictValue = (np.exp(input.T) / np.sum(np.exp(input), axis=1)).T

        return predictValue
    

