#When training a random forest model, we use sampling with replacement to draw randomly samples from the original sample set
#We will have a new set of samples, possibly with repeating samples
#For the XGBoost example, when we are drawing samples out of the orginal set, there is more possiblity that we pick the example that gets missclassified
#So XGBoost can make more suitable training sets to train a random forest model
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

#Reading the data file
df = pd.read_csv('heart.csv')
df.head()
print(df.head())

#One-hot encoding using Pandas, which means 
#to turn a multiclass or category variable into several binary 
#variables instead

#Puting category variables into a list
cat_variables = ['Sex', 'ChestPainType','RestingECG','ExerciseAngina','ST_Slope']

#get_dummies will help make the variables that has only two categories
#for the parameter columns, you input a list
#Use the parameter prefix to set the name of your 'dummies

df= pd.get_dummies(data=df, columns = cat_variables, prefix=cat_variables)
print(df.head)

#A dummy example: ST_Slope: Down, Flat, Up ---> ST-Slope_Down: True, False ......

#Seperate the data set into two parts, the features and target
#Put the name of features in a list

features = [x for x in df.columns if x not in ['HeartDisease']]
print(df.columns)
print(features)


#Split the X,y data in to two sets. The first set is for training.
#The second set is for validation or testing
#Use train_test_split from sklearn.model_selection
#Sklearn deals with data sheets from pandas, we need to pass the columns as X or y data
#Syntax for columns is dataframe[column_name]
#Use train_size parameter of 0.x so the examples in the first column returned has 0.x times
#all values
#Syntax we use for train_test_split: X_train, x_val, y_train, y_val = train_test_split(X_columns, y_columns,train_size = 0.x)
#0.x of data goes into X_train and y_train
#df[a_list_of_features]
X_train, X_val, y_train, y_val = train_test_split(df[features], df['HeartDisease'], train_size= 0.8)
print(len(X_train))
#XGBoost classfier takes an evaluation set of X and y to calculate the cost and use Gradient Descent to minimize the cost
#We need to have a evaluation set from the training set

X_train_fit, X_train_eval, y_train_fit, y_train_eval= train_test_split(X_train, y_train, train_size= 0.8)
print(len(X_train_fit))

#Use hyper parameter eval_set and pass a list of tuples to it. One tuple include X_train_eval and y_train_eval
#early_stopping_rounds means if there are x rounds where the cost does not decrease, stop the training
#Do not have too many training rounds, it leads to more estimators or trees and may cause overfitting

XGB_model = XGBClassifier(n_estimator = 300, learning_rate = 0.1)
XGB_model.fit(X_train_fit, y_train_fit, eval_set = [(X_train_eval,y_train_eval)], early_stopping_rounds = 10)
                          
accuracy_training = accuracy_score(XGB_model.predict(X_train), y_train)
accuracy_testing = accuracy_score(XGB_model.predict(X_val), y_val)
print(f'{accuracy_training}, {accuracy_testing}')
