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


#For DecisionTreeClassifier from sklearn.tree you have several hyper parameters
#min_sample_split: minimum examples you need for a split to happen in a node. Too small cause over fitting
#max_depth: the maximum depth you can have for your model. Too large causes overfitting

#Training the model on different min_sample_split value
min_samples_splits= [2,10,30,40,60,100,200,300,700]

accuracy_list_train = []
accuracy_list_val = []

#train the model and calculate accuracy stores
#we use DecisionTreeClassifier from sklearn.tree to define and fit a model
#Use hyper parameter min_samples_split in the DecisionTreeClassifier to set the minimum samples needed to split
#Syntax for constructing a deicision tree model and train it:
#model = DecisionTreeClassifier(min_samples_split = min_samples_split).fit(X_train, y_train)
#by creating a model object and use the fit function
#We use accuracy_score from sklearn.metrics to compare prediction and the original label so to get a accuracy score

for min_samples_split in min_samples_splits:
    model = DecisionTreeClassifier(min_samples_split= min_samples_split).fit(X_train, y_train)
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
#syntax plt.xticks(ticks = range(len(list_of_xlabels))), labels = list_of_xlabels
plt.xticks(ticks = range(len(min_samples_splits)), labels= min_samples_splits)
plt.show()

max_depth_list = [1,2,3,4,8,16,32,64]
max_depth_list2 = [1,2,3,4,8,16,32,54]

accuracy_list_train = []
accuracy_list_val = []

for max_depth in max_depth_list:
    model = DecisionTreeClassifier(max_depth= max_depth).fit(X_train, y_train)
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
plt.xticks(ticks=range(len(max_depth_list)), labels=max_depth_list)

plt.show()

#From the graph we have seen 40 min samples and a depth of 4 is the most optimal

#Set the model using the best hyper parameters possible
decision_tree_model= DecisionTreeClassifier(min_samples_split= 40, max_depth=4).fit(X_train, y_train)
train_accuracy= accuracy_score(decision_tree_model.predict(X_train), y_train)
val_accuracy = accuracy_score(decision_tree_model.predict(X_val), y_val)
print(train_accuracy, val_accuracy)
