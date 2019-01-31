import pandas as pd 
import sklearn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Adult data.csv is binary output of either making more than 50K or less 
enc_workclass = OneHotEncoder(handle_unknown='ignore')
enc_education = OneHotEncoder(handle_unknown='ignore')
enc_marital_status = OneHotEncoder(handle_unknown='ignore')
enc_occupation = OneHotEncoder(handle_unknown='ignore')
enc_relationship = OneHotEncoder(handle_unknown='ignore')

data = pd.read_csv("/Users/amitnarayan/Coding/Machine-Learning-Portfolio/Turicreate/adult.csv", error_bad_lines=False)
print(data)

"""le = sklearn.preprocessing.LabelEncoder()
X = data.apply(le.fit_transform)
print(X.head())"""

#print(type(enc_workclass))

#enc.fit(data["workclass"])
#enc.fit(data["workclass"])
#print(data)


X = data.iloc[:,:14]
Y = data.iloc[:, -1]
#print(type(Y))

#logisticRegression = LogisticRegression(solver="sag", penalty="l2")
#logisticRegression.fit(X,Y)