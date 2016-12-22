from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cross_validation import cross_val_score, KFold
import pickle as pk
import numpy as np
import os

from helpers import get_config




def _feature_generation(data):
    cat_columns = data.columns
    n_columns = len(cat_columns)

    for i in range(n_columns):
        for j in range(i + 1, n_columns):
            c1 = cat_columns[i]
            c2 = cat_columns[j]

            if len(data[c1].unique()) == 2 and len(data[c2].unique()) == 2:
                # print("XOR over %s and %s" % (c1, c2))
                yield data[c1] ^ data[c2], c1, c2


def feature_generation(data, c1=None, c2=None):
    if c1 is None or c2 is None:
        return _feature_generation(data)
    else:
        return data[c1] ^ data[c2]


class XORFeatureGeneration(BaseEstimator, TransformerMixin):
    def __init__(self, scorer=None, estimator=None, file=None, num_features=10):
        self.estimator = estimator
        self.scorer = scorer
        self.num_features = num_features
        self.xor_cats = {}
        self.file = file

    def load_xor_features(self, path):
        config = get_config()
        self.xor_cats = pk.load(open(os.path.join(config.DATA_DIR, "xor_cats.pk"), "rb"))

    def fit(self, X, y):

        if self.file:
            self.load_xor_features(self.file)

        if self.estimator:
            kf = KFold(X.shape[0], n_folds=5, shuffle=True)
            cv = cross_val_score(self.estimator, X, y, cv=kf, n_jobs=10, scoring=self.scorer)
            base_score = cv.mean()

        '''
        try:
            xor_cats = pk.load(open("data/xor_cats.pk", "rb"))
        except:
            xor_cats = {}
        '''

        self.xor_cats = {}

        for serie, c1, c2 in feature_generation(X):
            col = "xor_%s_%s" % (c1, c2)

            if not "%s_%s" % (c1, c2) in self.xor_cats:

                if self.estimator:
                    X[col] = serie

                    cv = cross_val_score(self.estimator, X, y, cv=kf,
                                         n_jobs=10, scoring=self.scorer)

                    self.estimator.fit(X, y)

                    score = cv.mean()

                    if score < base_score:
                        print("Col %s: %f" % (col, score))

                    del X[col]

                    self.xor_cats["%s_%s" % (c1, c2)] = score
            else:
                self.xor_cats["%s_%s" % (c1, c2)] = 0

        return self

    def transform(self, X):

        if self.file:
            self.load_xor_features(self.file)

        xor_cats = [(k, v) for k, v in self.xor_cats.items()]
        indexes = sorted(range(len(xor_cats)), key=lambda k: xor_cats[k][1])

        cols = []
        for ind in indexes[:20]:
            c, v = xor_cats[ind]
            X["xor_%s" % c] = feature_generation(X, *c.split("_"))

        return X

    def get_importance_matrix(self):

        min_value = min(self.xor_cats.values())
        max_value = max(self.xor_cats.values())

        columns = []
        for key in self.xor_cats.keys():
            f1, f2 = key.split("_")

            if f1 not in columns:
                columns.append(f1)

            if f2 not in columns:
                columns.append(f2)

        matrix = np.ones(shape=(len(columns), len(columns)), dtype=float)
        for i, f1 in enumerate(columns):
            for j in range(i+1, len(columns)):
                f2 = columns[j]

                key = "%s_%s" % (f1,f2)
                if key in list(self.xor_cats.keys()):
                    matrix[i, j] = (self.xor_cats[key] - min_value) / (max_value - min_value)
                    matrix[j, i] = matrix[i, j]

        return matrix

class KmeansFeatureGeneration(BaseEstimator, TransformerMixin):
    pass


class PCAFeatureGeneration(BaseEstimator, TransformerMixin):
    pass






