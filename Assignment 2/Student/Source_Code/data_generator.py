import numpy as np
import random 
from App.Pre_processing.data_generation import create_toy_data

if __name__ == "__main__":
    
    for i in range(10):        
        if i <5:
            train_x_data, train_y_data = create_toy_data()
            test_x_data, test_y_data = create_toy_data(training=False)
        else:
            train_x_data, train_y_data = create_toy_data(add_outliers=True)
            test_x_data, test_y_data = create_toy_data(add_outliers=True, training=False)

        np.savetxt("./Data/x_{}.csv".format(i), train_x_data)
        np.savetxt("./Data/y_{}.csv".format(i), train_y_data)
        np.savetxt("./Data/test_x_{}.csv".format(i), test_x_data)
        np.savetxt("./Data/test_y_{}.csv".format(i), test_y_data)
        


