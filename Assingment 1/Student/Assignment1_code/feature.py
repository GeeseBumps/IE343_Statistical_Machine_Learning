import numpy as np
import matplotlib.pyplot as plt
from App.polynomial import PolynomialFeatures

if __name__ == "__main__":
    x = np.linspace(-1, 1, 100)

    # User defined : only integer
    #feature_dimension = 1
    feature_dimension = np.random.randint(1, 9)

    X_polynomial = PolynomialFeatures(feature_dimension).transform(x[:, None])
    
    for j in range(12):
        plt.plot(x, X_polynomial[:, j])

    plt.savefig("./result.png")
