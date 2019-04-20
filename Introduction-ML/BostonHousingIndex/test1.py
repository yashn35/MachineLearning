
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression



iris = sns.load_dataset("iris")
Y_pred = lm.predict(X_test)
#print(iris.head())

y = iris.species
X = iris.drop('species',axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=100,
                                                    stratify=y)

x = tree.DecisionTreeClassifier()
x.fit(X_train,y_train)

y_pred = (x.predict(X_test))
print (accuracy_score(y_test, y_pred)* 100)
for i,j in zip(y_test,y_pred):
    print(i,j)
# http://scikit-learn.org/stable/modules/tree.html
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
