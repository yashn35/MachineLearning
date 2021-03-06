import pandas as pd
import sklearn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
 
'#Adultdata.csv is binary output of either making more than 50K or less'

file_path = "../../../Machine-Learning-Portfolio/Advanced-ML\
/LogisticRegression/adult.csv"

data = pd.read_csv(file_path, error_bad_lines=False)


'#This is to ensure original data doesnt get corrupted or changed'
data_v2 = data.append(data)


X_income = data_v2.copy()
X_income = data_v2.select_dtypes(include=['object'])
print("THIS IS X_INCOME", X_income)
X_enc = X_income.copy()

'#one-hot encoding values from my data_v2set using pd.getdummies method'
X_enc = pd.get_dummies(X_income, columns=['workclass', 'native-country',
                                          'gender', 'relationship', 'race',
                                          'occupation', 'workclass',
                                          'education', 'marital-status'])

'#Y_target Income is the target row or the output of the model'
Y_target = data_v2['income']

X_train, X_test, y_train, y_test = train_test_split(X_enc, Y_target, test_size=0.2)

logisticRegression = LogisticRegression(solver="sag", penalty="l2")
logisticRegression.fit(X_enc, Y_target)

Y_pred = logisticRegression.predict(X_test[0:10])
print(data_v2[0:10])  # Prints the original data; the ground truth
print(Y_pred)
'#Prints what the income the model predicted, either >50K or <= 50K'

'#Total Accuracy: Correct predictions / Total number of data points'
total_model_accuracy = logisticRegression.score(X_test, y_test)

print(total_model_accuracy)  # Model  accuracy is 83.3%'
"""#Additional Resources:
https://scikit-learn.org/stable/modules/generated/sklearn.
linear_model.LogisticRegression.html"""
