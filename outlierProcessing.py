import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

# Outlier/anomaly detection methods to be compared
from sklearn import svm
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Datasets
from sklearn.datasets import make_moons, make_blobs


class outlierProcessing(BaseEstimator, TransformerMixin):

    def __init__(self, outliers_fraction = 0.15, select_alg = 3):
        self.outliers_fraction = outliers_fraction
        
        self.select_alg = select_alg
        
        self.algorithms = [
            EllipticEnvelope(contamination=self.outliers_fraction ), \
            svm.OneClassSVM(nu=self.outliers_fraction, kernel="rbf", gamma=0.1), \
            IsolationForest(contamination=self.outliers_fraction, random_state=42), \
            LocalOutlierFactor(n_neighbors=35, contamination=self.outliers_fraction)
        ]        

        self.algorithm = self.algorithms[self.select_alg]
        #print(self.algorithm)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


# Example settings
n_samples = 300
n_outliers = int(0.15 * n_samples)
n_inliers = n_samples - n_outliers

blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
datasets = [make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5, **blobs_params)[0],
            make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5], **blobs_params)[0],
            make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3], **blobs_params)[0],
            4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] - np.array([0.5, 0.25])),
            14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)]

# Compare given classifiers under given settings
xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                     np.linspace(-7, 7, 150))

rng = np.random.RandomState(42)

X = datasets[0]

X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)

ol = outlierProcessing()
res = ol.fit_transform(X)


