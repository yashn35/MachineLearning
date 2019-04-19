# LogisticRegression-Voice.py uses logistic regression techniques to predict with a set of voice-data,
# and frequency of voice whether the person is a man or female.

import pandas as pd
import sklearn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# voice.csv is binary output of either Male or Female

file_path = "../../../Machine-Learning-Portfolio/Advanced-ML\
/LogisticRegression/voice.csv"

data = pd.read_csv(file_path, error_bad_lines=False)

'#This is to ensure original data doesnt get corrupted or changed'
data_v2 = data.append(data)
print(type(data_v2))

X_voice = data_v2.copy()
X_voice = data_v2.select_dtypes(include=['object'])
#print("THIS IS X_INCOME",X_voice)
X_enc = X_voice.copy()


'#one-hot encoding values from my data_v2set using pd.getdummies method'
#NOTE: you do not need to do one hot encoding on numbers
X_enc = data_v2['sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 
                            'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 
                            'maxfun', 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx']

"""X_enc = (X_voice, columns=['sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 
                            'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 
                            'maxfun', 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx'])"""

#sd	median	Q25	Q75	IQR	skew	kurt	sp.ent	sfm	mode	centroid	meanfun	minfun	maxfun	meandom	mindom	maxdom	dfrange	modindx

'#Y_target Income is the target row or the output of the model'
#Y_target = data_v2['label']

final_data = pd.concat([X_enc, Y_target], axis=1, sort=False)
"""pd.concat documentation
https://pandas.pydata_v2.org/pandas-docs/version/0.23.4/
generated/pandas.concat.html"""

X = final_data.drop(['label'], axis=1)
Y = final_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

logisticRegression = LogisticRegression(solver="sag", penalty="l2")
logisticRegression.fit(X, Y)

Y_pred = logisticRegression.predict(X_test[0:10])
print(data_v2[0:20])  # Prints the original data; the ground truth
print(Y_pred)
'#Prints what the income the model predicted, either >50K or <= 50K'

'#Total Accuracy: Correct predictions / Total number of data points'
total_model_accuracy = logisticRegression.score(X_test, y_test)

print(total_model_accuracy)  # Model  accuracy is 83.3%'
"""#Additional Resources:
https://scikit-learn.org/stable/modules/generated/sklearn.
linear_model.LogisticRegression.html"""

