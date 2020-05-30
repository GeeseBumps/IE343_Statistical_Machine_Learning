import numpy as np
import matplotlib.pyplot as plt

from App.data_import import filename_generation, filename_generation2, create_data_np
from App.decision_tree_maker import build_tree, predicts
from App.metric import gini_index
from App.evaluation import accruacy, average_metric
from App.plot import plot

from App.Pre_processing.polynomial import PolynomialFeatures

if __name__ == "__main__":
    synthetic = False
    
    if synthetic is True:
        filepath = "./Data/"
        file_name_func = filename_generation
    else:
        filepath = "./Data_2/"
        file_name_func = filename_generation2

    # For evaluation
    Accuracy_list = []

    # Is feature engineering used?
    is_feature = False
    feature = PolynomialFeatures(8) if is_feature == True else None
    
    for i in range(10):
        # Generatino filepath
        train_x, train_y, test_x, test_y = file_name_func(filepath, i)

        # File import
        train_x_data, train_y_data = create_data_np(train_x, train_y)
        train_y_data = train_y_data[:, None]
        test_x_data, test_y_data = create_data_np(test_x, test_y)
        
        # Feature engineering
        if is_feature:
            X_train = feature.transform(train_x_data)
            X_test = feature.transform(test_x_data)
        else:
            # No Feature engineering 
            X_train = train_x_data
            X_test = test_x_data

        # combine the features and targets
        X_train = np.hstack((X_train, train_y_data))
        

        if synthetic is True:
            #For plotting
            x1_test, x2_test = np.meshgrid(np.linspace(-5, 15, 100), np.linspace(-5, 15, 100))
            x_test_plot = np.array([x1_test, x2_test]).reshape(2, -1).T
        
            if is_feature == True:
                X_test_plot = feature.transform(x_test_plot)
            else:
                X_test_plot = x_test_plot
        
                
        # Set Hyperparameters
        max_depth = 5
        min_size = 10
        
        # Run algorithm (Training)
        tree = build_tree(X_train, max_depth, min_size, np.shape(X_train)[1]-1)

        # Predicting
        y_hat = predicts(tree, X_test)
        
        

        # Evaluating
        accruacy_value = accruacy(test_y_data, y_hat)

        # Appending
        Accuracy_list.append(accruacy_value)


        # Plotting
        if synthetic is True:
            y_hat_plot = predicts(tree, X_test_plot)
            plot(train_x_data, np.squeeze(train_y_data), test_x_data, test_y_data, x1_test, x2_test, y_hat_plot, "./Results/logits_result_{}".format(i))
        

    # average MSE 
    (average_accuracy, accuracy_std) = average_metric(Accuracy_list)

    # Print
    print(
        '[Average accuracy] \n',
        '{:7.4f} \n'.format(average_accuracy),
        '[accuracy_std] \n',
        '{:7.4f}'.format(accuracy_std))