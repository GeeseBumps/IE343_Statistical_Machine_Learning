import pandas as pd
import numpy as np

y=[2,1, 2, 1, 2, 0, 0, 0, 1, 0]
y=np.array(y)
one_hot_y = np.zeros((len(y), 3))
for i in range(len(y)):
    if y[i] == 0:
        one_hot_y[i][0] += 1
    elif y[i] == 1:
        one_hot_y[i][1] += 1
    else:
        one_hot_y[i][2] += 1
print(one_hot_y)