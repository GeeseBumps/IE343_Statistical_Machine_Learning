import numpy as np
import matplotlib.pyplot as plt

def plot(x_train, y_train, x_test, y_test, x1_test, x2_test, y_hat_plot, filename):
    plt.subplot(1,2,1)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=50, label="training data")
    plt.contourf(x1_test, x2_test, y_hat_plot.reshape(100, 100), alpha=0.1, levels=np.linspace(0, 1, 3))
    plt.xlim(-5, 15)
    plt.ylim(-5, 15)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.subplot(1,2,2)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=50, alpha=0.5, label="test data")
    plt.contourf(x1_test, x2_test, y_hat_plot.reshape(100, 100), alpha=0.1, levels=np.linspace(0, 1, 3))
    plt.xlim(-5, 15)
    plt.ylim(-5, 15)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig(filename + ".png")
    plt.clf()