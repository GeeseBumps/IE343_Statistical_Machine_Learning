import numpy as np

class Gaussian():
    """
    The Gaussian distribution
    p(x|mu, var)
    = exp{-0.5 * (x - mu)^2 / var} / sqrt(2pi * var)
    """

    def __init__(self, mu=None, var=None, tau=None):        
        self.mu = mu
        self.var = var
        
    # _fit method is to choose the parameters of guassian distribution according to a specific way. 
    # The reason why explicitly define "_fit" method is that you can choose various methods to define "mu" and "var" including maximum likelihood and bayesian method etc,. 
    def fit(self, X):
        self._ml(X)

    def _ml(self, X):
        self.mu = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)



    
