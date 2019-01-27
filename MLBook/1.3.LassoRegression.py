from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.datasets import load_boston
boston = load_boston()
X =boston.data[:,:]
y = boston.target
clf = Lasso()
clf.fit(X[:200,:4], y[:200])
print(clf.intercept_)

#print(clf.predict(X[480:,:4]))

"""
boson_data

Training

  1,2,3,4,5,6,7..13    target
1                        24
2                        13.2
3                        6
4
.
.
480

clf.fit(X,y)




Testing
481
.
.
.
.
.
506
"""
