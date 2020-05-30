import numpy as np
import pandas as pd
import math


class LogisticRegressor():
    def __init__(self,w=None):
        self.w = w

    def fit(self, X, y,lr,epoch_num):
        #TODO training
        self.w = np.zeros(np.size(X, 1))
        w = self.w
        ridge_lambda=np.exp(-3)
        for _ in range(epoch_num):
            w_prev = np.copy(w)
            y_hat = 1 / (1 + np.exp(-X @ w))
            grad = np.matmul((y_hat - y), X) + (ridge_lambda) * w_prev
            w -= lr * grad
            if np.allclose(w, w_prev):
                break
        self.w = w


    def predict(self, X):
        #TODO prediction
        predictValue = 1/(1+np.exp(-X @ self.w))


        return predictValue
