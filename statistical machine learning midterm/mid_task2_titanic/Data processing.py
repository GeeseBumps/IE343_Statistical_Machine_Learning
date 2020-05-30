import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series


train = pd.read_csv('./Data/titanic_train.csv', index_col=0)
test = pd.read_csv('./Data/titanic_test.csv', index_col=0)



#age의 평균값을 구하는 코드이다. 평균값을 구해 age에 존재하는 nan값을 채워줄 것이다.
age_sum=0
count = 0
for i in range(len(train['Age'])):
    age = train.iloc[i,4]
    if np.isnan(age):
        continue
    else:
        age_sum += age
        count +=1
avg_age=age_sum/count
print(age_sum,count,avg_age)



# 830,1,1,"Stone, Mrs. George Nelson (Martha Evelyn)",female,62.0,0,0,113572,80.0,B28,
#위에 있는 Embarked에 존재하는 nan 값을 채워주기 위해 Embarked의 mode를 사용할 것이다. 아래 코드는 mode를 조사하는 코드
train['Embarked'].replace('Q', 0, inplace=True)
train['Embarked'].replace('S', 1, inplace=True)
train['Embarked'].replace('C', 2, inplace=True)
count_Q= 0
count_S=0
count_C=0
for i in range(len(train['Survived'])):
    value = train.iloc[i, 10]
    if value==0:
        count_Q +=1
    elif value == 1:
        count_S+=1
    else:
        count_C+=1

print(count_Q,count_S,count_C)

#각각의 값은 (53 450 120)로 나왔으며 모드는 S이다. 따라서 S를 nan값에 넣어준다.

train['Initial'] = train.Name.str.extract('([A-Za-z]+)\.')
test['Initial'] = test.Name.str.extract('([A-Za-z]+)\.')

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

print(train.groupby('Initial').mean())