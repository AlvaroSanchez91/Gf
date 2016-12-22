import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
import numpy

from helpers import get_config

TREE_ESTIMATORS = {
    'extra_trees_regressor': ExtraTreesRegressor,
    'extra_trees_classifier': ExtraTreesClassifier,

}


class TreeBased(BaseEstimator, TransformerMixin):

    def __init__(self, tree_estimator, n_estimators, n_features):
        self.tree_estimator = tree_estimator
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.selected_features = []

    def fit(self, X, y):
        config = get_config()
        estimator = TREE_ESTIMATORS[self.tree_estimator](
            n_estimators=self.n_estimators,
            random_state=config.RANDOM_STATE,
        )

        estimator.fit(X, y)
        self.idx_sorted = np.argsort(-estimator.feature_importances_)


        if isinstance(X, DataFrame):
            self.selected_features = X.columns[self.idx_sorted[:self.n_features]]
        else:
            pass
        return self

    def transform(self, X):

        if isinstance(X, DataFrame):
            newX = X[self.selected_features]

        elif isinstance(X, numpy.ndarray):
            newX = X[:,self.idx_sorted[:self.n_features]]

        return newX
