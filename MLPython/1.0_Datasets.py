# Page 32, Introduction to Machine Learning: Some Sample Datasets

import mglearn
from matplotlib import pyplot as plt
X, y = mglearn.datasets.make_forge()
plt.scatter(X[:, 0], y)
plt.legend(["Class 0", "Class 1"], loc=0)
plt.xlabel("First Feature")
plt.ylabel("Second Feature")
plt.show()
print("X.shape {}".format(X.shape))



from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=5,centers=2, n_features=5,random_state=1)
from matplotlib import pyplot as plt
print(X[:,2],y)
plt.scatter(X[:, 2],y)
plt.legend(["Class 0", "Class 1"], loc=0)
plt.xlabel("First Feature")
plt.ylabel("Second Feature")
plt.show()