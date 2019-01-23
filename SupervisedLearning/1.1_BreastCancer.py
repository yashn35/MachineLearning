import sklearn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

"""
type
dir
print()
"""

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1,11)
for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test,y_test))
print(training_accuracy)
print(test_accuracy)

#How the X_train and y_train got plotted, mean accuracy, Dimeson

"""
print(cancer.keys())
print(cancer.data.shape)
print(cancer.target_names, np.bincount(cancer.target))
print("Test".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print(cancer.DESCR)
"""





# for i in cancer.keys():
#     print (cancer[i])

print(cancer.target_names)