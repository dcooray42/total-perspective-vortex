import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from CSP import CommonSpatialPatterns

class CSPPairwise(BaseEstimator, TransformerMixin) :
    def __init__(self, class_pairs, n_components=4, log=True):
        self.class_pairs = class_pairs
        self.n_components = n_components
        self.log=log
        self.csp_list = []

    def fit(self, X, y) :
        for class_a, class_b in self.class_pairs :
            class_indices = np.where(np.logical_or(y == class_a, y == class_b))[0]
            subset_X = X[class_indices]
            subset_y = y[class_indices]
            
            csp = CommonSpatialPatterns(n_components=self.n_components, log=self.log)
            csp.fit(subset_X, subset_y)
            self.csp_list.append(csp)
        
        return self

    def transform(self, X) :
        csp_features_combined = []
        for csp in self.csp_list :
            csp_features = csp.transform(X)
            csp_features_combined.append(csp_features)
        
        return np.concatenate(csp_features_combined, axis=1)
