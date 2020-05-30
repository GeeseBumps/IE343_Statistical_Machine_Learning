import numpy as np

def binary_cross_entropy(y, y_hat):
    BCE = -1 * np.mean(y * np.log(y_hat + 1e-7) + (1-y) * np.log(1 - y_hat + 1e-7))
    return BCE

def average_BCE(BCE_list):
    average_BCE = np.mean(BCE_list)
    std_BCE = np.sqrt(np.mean(np.square(BCE_list - average_BCE)))

    return (average_BCE, std_BCE)