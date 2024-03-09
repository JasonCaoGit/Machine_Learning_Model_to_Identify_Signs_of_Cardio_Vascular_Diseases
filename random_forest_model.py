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

#Next, we will train a random forest model to predict the result
#This algotihm has three hyper parameters: min_samples_split, max_depth, n_estimators
#n-estimators mean how many decision tree you would like to have in your random forest model

min_samples_splits = [2,10,30,40,60, 100,200,300, 700]
accuracy_list_train= []
accuracy_list_val = []
for min_samples_split in min_samples_splits:
    model = RandomForestClassifier(min_samples_split= min_samples_split).fit(X_train, y_train)
    predictions_train= model.predict(X_train)
    predictions_val= model.predict(X_val)
    accuracy_train= accuracy_score(predictions_train, y_train)
    accuracy_val= accuracy_score(predictions_val, y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

print(accuracy_list_train)
print(accuracy_list_val)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.xticks(ticks = range(len(min_samples_splits)), labels= min_samples_splits)
plt.show()


max_depth_list = [2,4,8,16,32,64]
accuracy_list_train= []
accuracy_list_val = []
for max_depth in max_depth_list:
    model = RandomForestClassifier(max_depth= max_depth).fit(X_train, y_train)
    predictions_train= model.predict(X_train)
    predictions_val= model.predict(X_val)
    accuracy_train= accuracy_score(predictions_train, y_train)
    accuracy_val= accuracy_score(predictions_val, y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

print(accuracy_list_train)
print(accuracy_list_val)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.xticks(ticks = range(len(max_depth_list)), labels= max_depth_list)
plt.show()


n_estimators_list = [10, 50,100, 200, 300]
accuracy_list_train= []
accuracy_list_val = []
for n_estimators in n_estimators_list:
    model = RandomForestClassifier(n_estimators= n_estimators).fit(X_train, y_train)
    predictions_train= model.predict(X_train)
    predictions_val= model.predict(X_val)
    accuracy_train= accuracy_score(predictions_train, y_train)
    accuracy_val= accuracy_score(predictions_val, y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

print(accuracy_list_train)
print(accuracy_list_val)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.xticks(ticks = range(len(n_estimators_list)), labels= n_estimators_list)
plt.show()

#best min_samples_split = 30; max_depth = 8,16; n_estimators = 100
random_forest_model = RandomForestClassifier(max_depth=16, min_samples_split= 30,n_estimators= 100).fit(X_train, y_train)

train_accuracy= accuracy_score(random_forest_model.predict(X_train), y_train)
val_accuracy = accuracy_score(random_forest_model.predict(X_val), y_val)
print(train_accuracy, val_accuracy)






