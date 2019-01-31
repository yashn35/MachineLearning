import pandas as pd 
import sklearn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Adultdata.csv is binary output of either making more than 50K or less

data = pd.read_csv("/Users/amitnarayan/Coding/Machine-Learning-Portfolio/Turicreate/adult.csv", error_bad_lines=False)

data_v2 = data.append(data) #I do this  in order to make sure "data" on line 11 doesn't get corrupted or changed in some way

X_income = data_v2.copy() 
X_income = data_v2.select_dtypes(include=['object'])
X_enc = X_income.copy()
X_enc = pd.get_dummies(X_income, columns=['workclass','native-country', 'gender','relationship', 'race', 'occupation', 'workclass','education','marital-status']) #one-hot encoding values from my data_v2set using pd.getdummies method 
Y_target = data_v2['income'] #'Income' is the target row or the output of the model
final_data = pd.concat([X_enc, Y_target], axis = 1, sort=False) #pd.concat documentation https://pandas.pydata_v2.org/pandas-docs/version/0.23.4/generated/pandas.concat.html

X = final_data.drop(['income'], axis = 1)
Y = final_data['income'] 
 
X_train, X_test, y_train,y_test = train_test_split(X, Y, test_size=0.2)

logisticRegression = LogisticRegression(solver="sag", penalty="l2")
logisticRegression.fit(X,Y)

Y_pred = logisticRegression.predict(X_test[0:10])
print(data_v2[0:10]) #This line prints the original data; the ground truth
print (Y_pred) #This line prints what the income the model predicted, either >50K or <= 50K

score = logisticRegression.score(X_test, y_test) #Total Accuracy; Correct predictions / Total number of data points 
print (score) #Model total accuracy = 83.3%

#Additional Resources: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html