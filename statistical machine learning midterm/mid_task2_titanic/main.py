import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from App.logistic_regressor import LogisticRegressor
from pandas import Series

def getData():
    # load and preprocess
    train = pd.read_csv('./Data/titanic_train.csv', index_col=0)
    test = pd.read_csv('./Data/titanic_test.csv', index_col=0)

    # TODO
    #bias term
    train['bias'] = 1
    test['bias'] = 1

    # 남자는 0, 여자는 1을 부여한다.
    train['Sex'].replace('male', 0, inplace=True)
    train['Sex'].replace('female', 1, inplace=True)
    test['Sex'].replace('male', 0, inplace=True)
    test['Sex'].replace('female', 1, inplace=True)

    # initial column 추가
    train['Initial'] = train.Name.str.extract('([A-Za-z]+)\.')
    test['Initial'] = test.Name.str.extract('([A-Za-z]+)\.')
    train['Initial'].replace(
        ['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don',
         'Dona'],
        ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr', 'Mr'],
        inplace=True)

    test['Initial'].replace(
        ['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don',
         'Dona'],
        ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr', 'Mr'],
        inplace=True)

    # 혼자 탑승했으면 TravelAlone이 1 아니면 0을  feature로 추가한다.
    #train['TravelAlone'] = np.where((train["SibSp"] + train["Parch"]) > 0, 0, 1)
    #test['TravelAlone'] = np.where((test["SibSp"] + test["Parch"]) > 0, 0, 1)

    #Family Size를 feature로 추가한다.
    #train['FamilySize'] = train['SibSp']+train['Parch']+1
    #test['FamilySize'] = test['SibSp']+test['Parch']+1


    # Age에 비어있는 null 값을 전체 나이의 평균값으로 넣어준다. 평균값 구하는 코드는 Data processing.py에 있다.

    train.loc[(train.Age.isnull()) & (train.Initial == 'Mr'), 'Age'] = 33
    train.loc[(train.Age.isnull()) & (train.Initial == 'Mrs'), 'Age'] = 36
    train.loc[(train.Age.isnull()) & (train.Initial == 'Master'), 'Age'] = 5
    train.loc[(train.Age.isnull()) & (train.Initial == 'Miss'), 'Age'] = 22
    train.loc[(train.Age.isnull()) & (train.Initial == 'Other'), 'Age'] = 46

    test.loc[(test.Age.isnull()) & (test.Initial == 'Mr'), 'Age'] = 33
    test.loc[(test.Age.isnull()) & (test.Initial == 'Mrs'), 'Age'] = 36
    test.loc[(test.Age.isnull()) & (test.Initial == 'Master'), 'Age'] = 5
    test.loc[(test.Age.isnull()) & (test.Initial == 'Miss'), 'Age'] = 22
    test.loc[(test.Age.isnull()) & (test.Initial == 'Other'), 'Age'] = 46
    '''
    train.loc[train.Age.isnull(), 'Age'] = 30
    test.loc[test.Age.isnull(), 'Age'] = 30
    '''

    # Name, Cabin, Ticket 등 regression에 필요하지 않은 feature들을 제거해준다.
    train_X = train.drop(['Name', 'Cabin', 'Ticket', 'Survived', 'Embarked','Initial','Fare'], 1).values
    test_X = test.drop(['Name', 'Cabin', 'Ticket', 'Survived', 'Embarked','Initial','Fare'], 1).values


    # test_y, train_y 를 지정해주고 train_X와 test_X에서 survived 열을 제거한다.

    test_y = test['Survived'].values
    train_y = train['Survived'].values

    return train_X, train_y, test_X, test_y


def accuracy(true_y, pred_y):
    pred_y[pred_y < 0.5] = 0
    pred_y[pred_y >= 0.5] = 1
    accuracy = np.sum(true_y == pred_y) / len(true_y) * 100
    print('Accuracy', accuracy, "%")
    f = open('./result/test_Accuracy.txt', 'w')
    f.write(str(accuracy))
    f.close()


if __name__ == "__main__":
    train_X, train_y, test_X, test_y = getData()

    # Model
    model = LogisticRegressor()
    lr = 0.001  # TODO #learning rate
    epoch = 10000  # TODO #epoch number
    # Training
    model.fit(train_X, train_y, lr, epoch)

    # Prediction
    pred_y = model.predict(test_X)
    # Evaluation
    accuracy(test_y, pred_y)