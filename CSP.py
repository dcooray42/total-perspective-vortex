import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin

class CommonSpatialPatterns(TransformerMixin, BaseEstimator) :
    def __init__(self, n_components=4, log=None) :
        if not isinstance(n_components, int) :
            raise ValueError("n_components must be an integer.")
        self.n_components = n_components
        self.log = log

    def _check_Xy(self, X, y=None) :
        if not isinstance(X, np.ndarray) :
            raise ValueError("X should be of type ndarray (got %s)." % type(X))
        if y is not None:
            if len(X) != len(y) or len(y) < 1:
                raise ValueError("X and y must have the same length.")
        if X.ndim < 3:
            raise ValueError("X must have at least 3 dimensions.")

    def fit(self, X, y) :
        self._check_Xy(X, y)

        self._classes = np.unique(y)
        n_classes = len(self._classes)
        if n_classes != 2 :
            raise ValueError("n_classes must be equals 2.")

        covs = self._compute_covariance_matrices(X, y)
        eigen_values, eigen_vectors = linalg.eigh(covs[0], covs.sum(0))
        ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]

        eigen_vectors = eigen_vectors[:, ix]

        self.filters_ = eigen_vectors.T

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        X = (X**2).mean(axis=2)

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

    def transform(self, X) :
        if not isinstance(X, np.ndarray) :
            raise ValueError(f"X should be of type ndarray (got {type(X)}).")
        if self.filters_ is None:
            raise RuntimeError(
                "No filters available. Please first fit CSP " "decomposition."
            )

        pick_filters = self.filters_[: self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        X = (X**2).mean(axis=2)
        log = True if self.log is None else self.log
        if log:
            X = np.log(X)
        else:
            X -= self.mean_
            X /= self.std_
        return X
    
    def _compute_covariance_matrices(self, X, y) :
        _, n_channels, _ = X.shape
        covs = []

        for this_class in self._classes:
            x_class = X[y == this_class]
            x_class = np.transpose(x_class, [1, 0, 2])
            x_class = x_class.reshape(n_channels, -1)
            covar_matrix = self._calc_covariance(x_class)
            covs.append(covar_matrix)
        return np.stack(covs)
    
    def _calc_covariance(self, X, ddof=0) :
        X -= X.mean(axis=1)[:, None]
        N = X.shape[1]
        return np.dot(X, X.T.conj()) / float(N - ddof)