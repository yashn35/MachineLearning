# LogisticRegression-Advertising.py uses logistic regression techniques to predict with
# a set of ad-data whether someone clicked on the ad or not
# Data was taken from this Kaggle link: https://www.kaggle.com/fayomi/advertising/version/1

import pandas as pd
import sklearn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

file_path = "../../../Machine-Learning-Portfolio/Advanced-ML\
/LogisticRegression/advertising.csv"

data = pd.read_csv(file_path, error_bad_lines=False)

'#This is to ensure original data doesnt get corrupted or changed'
data_v2 = data.append(data)

X_advertising = data_v2.copy()
X_advertising = data_v2.select_dtypes(include=['object'])
X_enc = X_advertising.copy()

'#one-hot encoding values from my data_v2set using pd.getdummies method'
X_enc_1 = pd.get_dummies(X_advertising, columns=['Ad Topic Line', 'City', 'Country', 'Timestamp'])
#Variable for all the data that does not need to one-hot econded
X_enc_2 = X_advertising.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1)

'#Y_target Income is the target row or the output of the model'
Y_target = data_v2['Clicked on Ad']

final_data = pd.concat([X_enc_1, X_enc_2, Y_target], axis=1, sort=False)
#print("FINAL DATA:", final_data)
"""pd.concat documentation
https://pandas.pydata_v2.org/pandas-docs/version/0.23.4/
generated/pandas.concat.html"""

X = final_data.drop(['Clicked on Ad'], axis=1)
Y = final_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

logisticRegression = LogisticRegression(solver="sag", penalty="l2")
logisticRegression.fit(X, Y)

Y_pred = logisticRegression.predict(X_test[0:20])
print(data_v2[0:20])  # Prints the original data; the ground truth
print(Y_pred)
'#Prints whether they clicked on the ad or not based on the data'

'#Total Accuracy: Correct predictions / Total number of data points'
total_model_accuracy = str(logisticRegression.score(X_test, y_test) * 100) + str("% accuracy")
print(total_model_accuracy)  
"""Additional Resources:
https://scikit-learn.org/stable/modules/generated/sklearn.
linear_model.LogisticRegression.html"""