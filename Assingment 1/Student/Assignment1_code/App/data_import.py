import csv
import numpy as np

def create_data(filepath1, filepath2):
    f1 = open(filepath1, 'r', encoding='utf-8')
    f2 = open(filepath2, 'r', encoding='utf-8')

    rdr1 = csv.reader(f1)
    rdr2 = csv.reader(f2)

    x = []
    t = []

    for line in rdr1:
        x = np.array(line)

    for line in rdr2:
        t = np.array(line)
    
    return x, t

def create_data_np(filepath1, filepath2):
    
    x = np.genfromtxt(filepath1, delimiter=',', dtype=None, encoding='UTF-8')
    t = np.genfromtxt(filepath2, delimiter=',', dtype=None, encoding='UTF-8')

    return x, t

def filename_generation(path, i):
    train_x_filepath = path + "x_{}.csv".format(i) 
    train_y_filepath = path + "y_{}.csv".format(i)
    test_x_filepath = path + "test_x_{}.csv".format(i)
    test_y_filepath = path + "test_y_{}.csv".format(i)

    return train_x_filepath, train_y_filepath, test_x_filepath, test_y_filepath

    
if __name__ == "__main__":
    filepath = "./Data/"

    train_x, train_y, test_x, test_y = filename_generation(filepath, 1)

    # File import
    train_x_data, train_y_data = create_data(train_x, train_y)

    print(train_x_data)
