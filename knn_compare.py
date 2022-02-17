import knn
import numpy as np
import pandas as pd
from pandas import testing
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets


# load iris
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')
d = {i: category for i, category in enumerate(iris.target_names)}
y = y.apply(lambda x: d[x])

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# knn.py
k, dist = 3, 'euclidean'
est = knn.KNNClassifier(k=k, dist=dist)
est.fit(X_train, y_train)
print('knn score: ', est.score(X_test, y_test))

# sklearn
est_sklearn = KNeighborsClassifier(n_neighbors=k, metric=dist)
est_sklearn.fit(X_train, y_train)
print('sklearn score: ', est_sklearn.score(X_test, y_test))

# compare test predictions
testing.assert_series_equal(est.predict(X_test), pd.Series(est_sklearn.predict(X_test)))