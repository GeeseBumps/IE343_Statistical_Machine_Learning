import numpy as np
from App.regressor import Regressor


class LinearRegressor(Regressor):
    """
    Linear regression model
    y = X @ w
    t ~ N(t|X @ w, var), we don't need to define var. 
    """

    def _fit(self, X, t):
        '''
        w is the least square method.
        '''
        self.w = 
        
    def _predict(self, X, return_std=False):
        y = X @ self.w        

        return y
