from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import mglearn
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
ridge = Ridge().fit(X_train, y_train)
print(ridge.coef_)