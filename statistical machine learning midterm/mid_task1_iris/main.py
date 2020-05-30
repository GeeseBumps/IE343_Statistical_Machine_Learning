import pandas as pd
import numpy as np
#from App.logistic_regressor import LogisticRegressor
from App.logistic_regressor import LogisticRegressor


def getData():
    train=pd.read_csv('./Data/iris_train.csv',index_col=0)
    test=pd.read_csv('./Data/iris_test.csv',index_col=0)
    
    train['Species'].replace('Iris-virginica',0,inplace=True)
    train['Species'].replace('Iris-setosa',1,inplace=True)
    train['Species'].replace('Iris-versicolor',2,inplace=True)
    test['Species'].replace('Iris-virginica',0,inplace=True)
    test['Species'].replace('Iris-setosa',1,inplace=True)
    test['Species'].replace('Iris-versicolor',2,inplace=True)
    
    train_y=train['Species'].values
    train_X=train.drop('Species',1).values
    train_X = np.insert(train_X, 0, 1, axis=1)
    test_y=test['Species'].values
    test_X=test.drop('Species',1).values
    test_X=np.insert(test_X, 0, 1, axis=1)

    return train_X,train_y,test_X,test_y


def accuracy(true_y,pred_y):
    predict=np.argmax(pred_y, axis=1)
    accuracy=np.sum(true_y==predict)/len(true_y)*100
    print('test_Accuracy', accuracy,"%" )
    f = open('./result/test_Accuracy.txt', 'w')
    f.write(str(accuracy))
    f.close()
    
    


if __name__ == "__main__":

    train_X, train_y, test_X, test_y = getData()
    # Model
    model = LogisticRegressor()
    lr =  0.001
    epoch = 5000
    # Training (Learning)
    model.fit(train_X, train_y, lr, epoch)
    # Prediction
    pred_y = model.predict(test_X)
    print(pred_y)
    # Evaluation
    accuracy(test_y, pred_y)

        
        


