from sklearn.linear_model import LinearRegression
import mglearn
from sklearn.model_selection import train_test_split
X,y = mglearn.datasets.make_wave(n_samples=60)
#print((mglearn.datasets.make_wave.keys))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
print(lr.coef_)
