import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

class outlierProcessing(BaseEstimator, TransformerMixin):

    def __init__(self, ):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X**3
        return X




X = np.array([1,2,3,4,5])

ol = outlierProcessing()
res = ol.fit_transform(X)

print(res)

