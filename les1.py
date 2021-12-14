import pandas as pd
import numpy as np
np.set_printoptions(precision=2)


df = pd.read_csv('data/titanic_train.csv', index_col='PassengerId')
print(df.head())
print(df.describe())
print(df[(df['Embarked'] == 'C') & (df['Fare'] > 200)].sort_values(by='Fare',ascending=False).transpose().head(10))
print(type(df))
print(type(df['Fare']))

print(pd.crosstab(df['Sex'],df['Survived']))
print(df[df['Sex'] == 'female']['Sex'].count(),df[df['Sex'] == 'male']['Sex'].count() )
print(pd.crosstab(df['Pclass'],df['Sex']))
print(df['Fare'].describe())

# print(df['Age'])
def old(age):
    if age < 30:
        return 1
    if age > 60:
        return 2

df['Old'] = df['Age'].apply(old)



print(df[df['Age'] < 30]['Age'].count())
print(pd.crosstab(df['Survived'],df['Old']))

print(pd.crosstab(df['Survived'],df['Old']))
