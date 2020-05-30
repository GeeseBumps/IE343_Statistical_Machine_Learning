import numpy as np
from App.Pre_processing.classifier import Classifier
from App.Pre_processing.label_transformer import LabelTransformer


class LeastSquaresClassifier(Classifier):
    """
    Least squares classifier model

    X : (N, D)
    W : (D, K)
    y = argmax_k X @ W
    """

    def __init__(self, W=None):
        self.W = W

    def _fit(self, X, t):
        """
        least squares fitting for classification

        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        t : (N,) or (N, K) np.ndarray
            training dependent variable
            in class index (N,) or one-of-k coding (N,K)
        """

        # One-hot encoding
        if t.ndim == 1:
            t = LabelTransformer().encode(t)

        self.W = np.linalg.pinv(X) @ t

    def classify(self, X:np.ndarray):
        """
        classify input data

        Parameters
        ----------
        X : (N, D) np.ndarray
            independent variable to be classified

        Returns
        -------
        (N,) np.ndarray
            class index for each input
        """
        
        return np.argmax(X @ self.W, axis=-1)
