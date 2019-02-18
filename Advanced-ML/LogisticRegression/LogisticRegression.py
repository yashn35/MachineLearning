import pandas as pd
import sklearn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

'#Adultdata.csv is binary output of either making more than 50K or less'

<<<<<<< HEAD
data = pd.read_csv("../../../Machine-Learning-Portfolio/Turicreate/adult.csv", error_bad_lines=False)
=======
file_path = "../../../Machine-Learning-Portfolio/Advanced-ML\
/LogisticRegression/adult.csv"
>>>>>>> bf12d241d668c67af53ae2a248d7dae6abb82d13

data = pd.read_csv(file_path, error_bad_lines=False)

'#This is to ensure original data doesnt get corrupted or changed'
data_v2 = data.append(data)


X_income = data_v2.copy()
X_income = data_v2.select_dtypes(include=['object'])
X_enc = X_income.copy()

<<<<<<< HEAD
X = final_data.drop(['income'], axis = 1)
Y = final_data['income'] 
 
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
=======
'#one-hot encoding values from my data_v2set using pd.getdummies method'
X_enc = pd.get_dummies(X_income, columns=
    ['workclass', 'native-country', 'gender', 'relationship',
        'race', 'occupation', 'workclass', 'education', 'marital-status'])

'#Y_target Income is the target row or the output of the model'
Y_target = data_v2['income']

final_data = pd.concat([X_enc, Y_target], axis=1, sort=False)
"""pd.concat documentation
https://pandas.pydata_v2.org/pandas-docs/version/0.23.4/generated/pandas.concat.html"""

X = final_data.drop(['income'], axis=1)
Y = final_data['income']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
>>>>>>> bf12d241d668c67af53ae2a248d7dae6abb82d13

logisticRegression = LogisticRegression(solver="sag", penalty="l2")
logisticRegression.fit(X, Y)

Y_pred = logisticRegression.predict(X_test[0:10])
print(data_v2[0:10]) '#Prints the original data; the ground truth'
print (Y_pred)  '#Prints what the income the model predicted, either >50K or <= 50K'

'#Total Accuracy: Correct predictions / Total number of data points'
total_model_accuracy = logisticRegression.score(X_test, y_test)

print (total_model_accuracy)  '#Model  accuracy is 83.3%'
"""#Additional Resources:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"""
