import numpy as np

def gradient_descent(w, grad, learning_rate):
    w -= learning_rate * grad
    return w