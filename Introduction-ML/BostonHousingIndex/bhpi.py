import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


sns.set_style("whitegrid")
sns.set_context("poster")

boston = load_boston()
b = pd.DataFrame(boston.data)
b.columns = boston.feature_names

b['PRICE'] = boston.target

X = b.drop('PRICE', axis = 1)
Y = b['PRICE']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)






lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)
print("Mean squared error: %.2f"% sklearn.metrics.mean_squared_error(Y_test,Y_pred))

