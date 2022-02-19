import pca
import numpy as np
import pandas as pd
from pandas import testing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import datasets


# load iris
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')
d = {i: category for i, category in enumerate(iris.target_names)}
y = y.apply(lambda x: d[x])

# pca.py
n_components = 0.9
est = pca.PCA(n_components)
est.fit(X)

# sklearn
est_sklearn = PCA(n_components)
est_sklearn.fit(X)

# compare
testing.assert_frame_equal(
    est.transform(X).abs(), 
    pd.DataFrame(est_sklearn.transform(X)).abs()
)
testing.assert_frame_equal(
    est.inverse_transform(est.transform(X)), 
    pd.DataFrame(est_sklearn.inverse_transform(est_sklearn.transform(X)))
)