import numpy as np
from App.Pre_processing.classifier import Classifier
from App.Pre_processing.optimizer import gradient_descent

class LogisticRegression(Classifier):
    """
    Logistic regression model

    y = sigmoid(X @ w)
    t ~ Bernoulli(t|y)
    """

    def __init__(self, w=None):
        self.w = w  
    
    def _sigmoid(self, a):
        return 1 / (1 + np.exp(-1 * a))

    def _fit(self, X, t, max_iter=100):
        """
        maximum likelihood estimation of logistic regression model

        Parameters
        ----------
        X : (N, D) np.ndarray
            training data independent variable
        t : (N,) np.ndarray
            training data dependent variable
            binary 0 or 1
        max_iter : int, optional
            maximum number of paramter update iteration (the default is 100)
        """
        
        self.w = np.zeros(np.size(X, 1))
        w = self.w

        for _ in range(max_iter):
            w_prev = np.copy(w)
            y_hat = self.proba(X)
            grad = np.matmul((y_hat-t),X)
            w = gradient_descent(w, grad, learning_rate=0.1)
            if np.allclose(w, w_prev):
                break

        self.w = w

    def proba(self, X):
        """
        compute probability of input belonging class 1

        Parameters
        ----------
        X : (N, D) np.ndarray
            training data independent variable

        Returns
        -------
        (N,) np.ndarray
            probability of positive
        """
        return self._sigmoid(X @ self.w)

    def classify(self, X, threshold=0.5):
        """
        classify input data

        Parameters
        ----------
        X : (N, D) np.ndarray
            independent variable to be classified
        threshold : float, optional
            threshold of binary classification (default is 0.5)

        Returns
        -------
        (N,) np.ndarray
            binary class for each input
        """
        return (self.proba(X) > threshold).astype(np.int)
