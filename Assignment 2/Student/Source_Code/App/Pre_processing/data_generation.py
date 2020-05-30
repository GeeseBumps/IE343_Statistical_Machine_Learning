import numpy as np

def create_toy_data(add_outliers=False, add_class=False, training=True):
    # x0 \in N(-1, 1) + \xi
    # X1 \in N(1, 1) + \xi
    if training:
        length = 50
    else:
        length = 20


    x0 = np.random.normal(size=length).reshape(-1, 2) - 1
    x1 = np.random.normal(size=length).reshape(-1, 2) + 1.

    if add_outliers:
        x_1 = np.random.normal(size=np.int(length/5)*2).reshape(-1, 2) + np.array([5., 10.]) #braodcasting
        return np.concatenate([x0, x1, x_1]), np.concatenate([np.zeros(np.int(length/2)), np.ones(np.int(length/2) + np.int(length/5))]).astype(np.int)
    if add_class:
        x2 = np.random.normal(size=length).reshape(-1, 2) + 3.
        return np.concatenate([x0, x1, x2]), np.concatenate([np.zeros(length/2.), np.ones(length/2.), 2 + np.zeros(length/2.)]).astype(np.int)

    return np.concatenate([x0, x1]), np.concatenate([np.zeros(np.int(length/2)), np.ones(np.int(length/2))]).astype(np.int)
    
    