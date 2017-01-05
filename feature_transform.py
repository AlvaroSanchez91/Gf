import numpy as np
import pandas as pd

#esto es mio
import pickle as pk
import os
from helpers import  get_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
#clf = ensemble.GradientBoostingRegressor()
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
#clf = svm.SVC()
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from helpers import  split_data

from sklearn.cluster import KMeans
#hasta aqui

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, Imputer


class DummyId(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X):

        X['id'] = range(X.shape[0])

        return X

class Elimina_Filas(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        X=X.drop(X.index[list(range(900,160000))])

        return X

class GuardaDatosProcesados(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, nombre="cs570con20colum.csv"):
        self.columns = columns
        self.nombre=nombre

    def fit(self, X, y):
        return self

    def transform(self, X):
        config = get_config()
        #np.savetxt("C:\\Users\\Usuario\Desktop\cs570con20colum.csv", X, delimiter=",")
        data=pd.DataFrame(X).copy()
        #data.to_csv(os.path.join(config.DATA_DIR, "processed.csv"), index=False)
        data.to_csv(self.nombre, index=False)

        return X

class ModelStakingLevel1(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns
        #self.rf_algoritmo=None
        self.my_rf_algoritm=None
        self.my_GradientBoostingRegressor_algoritm=None
        self.my_AdaBoostClassifier_algoritm=None
        self.my_SVC_algoritm=None
        self.my_BaggingClassifier_algoritm=None
        self.my_LogisticRegression_algoritm=None


    def fit(self, X, y):
        #self.rf_algoritmo=RandomForestClassifier().fit(X,y).predict
        self.my_rf_algoritm=RandomForestClassifier().fit(X,y).predict
        self.my_GradientBoostingRegressor_algoritm=GradientBoostingRegressor().fit(X,y).predict
        self.my_AdaBoostClassifier_algoritm=AdaBoostClassifier().fit(X,y).predict
        self.my_SVC_algoritm=SVC().fit(X,y).predict
        self.my_BaggingClassifier_algoritm=BaggingClassifier().fit(X,y).predict
        self.my_LogisticRegression_algoritm=LogisticRegression().fit(X,y).predict


        return self

    def transform(self, X):
        #model_rf = pk.load(open("cs570/models\\rf_0.32241.pk", "rb"))
        #X_sin_y=X.drop('y', axis=1,inplace=True).copy()
        #predict = model_rf.predict(X_sin_y)
        #X['predict_rf'] = predict

        #X_guardada=pd.read_csv('cs570con20colum.csv')
        #y_guardada=pd.read_csv('Ycs570.csv')
        X2=pd.DataFrame(X)
        #predicciones_rf=self.rf_algoritmo(X)
        prediction_rf=self.my_rf_algoritm(X)
        prediction_GradientBoostingRegressor=self.my_GradientBoostingRegressor_algoritm(X)
        prediction_AdaBoostClassifier=self.my_AdaBoostClassifier_algoritm(X)
        prediction_SVC=self.my_SVC_algoritm(X)
        prediction_BaggingClassifier=self.my_BaggingClassifier_algoritm(X)
        prediction_LogisticRegression=self.my_LogisticRegression_algoritm(X)


        X2['prediction_rf'] = prediction_rf
        X2['prediction_GradientBoostingRegressor'] = prediction_GradientBoostingRegressor
        X2['prediction_AdaBoostClassifier'] = prediction_AdaBoostClassifier
        X2['prediction_SVC'] = prediction_SVC
        X2['prediction_BaggingClassifier'] = prediction_BaggingClassifier
        X2['prediction_LogisticRegression'] = prediction_LogisticRegression


        return X2

class ModelStakingLevel1_regresor(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.my_rf_algoritm=None
        self.my_GradientBoostingRegressor_algoritm=None
        self.my_ElasticNet_algoritm=None
        self.my_linearRegressor_algoritm=None
        self.my_XGBregresor_algoritm=None
        self.my_KNN_algoritm=None


    def fit(self, X, y):
        self.my_rf_algoritm=RandomForestRegressor().fit(X,y).predict
        self.my_GradientBoostingRegressor_algoritm=GradientBoostingRegressor().fit(X,y).predict
        self.my_ElasticNet_algoritm=ElasticNet().fit(X,y).predict
        self.my_linearRegressor_algoritm=LinearRegression().fit(X,y).predict
        self.my_XGBregresor_algoritm=XGBRegressor().fit(X,y).predict
        self.my_KNN_algoritm=KNeighborsRegressor().fit(X,y).predict


        return self

    def transform(self, X):
        #model_rf = pk.load(open("cs570/models\\rf_0.32241.pk", "rb"))
        #X_sin_y=X.drop('y', axis=1,inplace=True).copy()
        #predict = model_rf.predict(X_sin_y)
        #X['predict_rf'] = predict

        #X_guardada=pd.read_csv('cs570con20colum.csv')
        #y_guardada=pd.read_csv('Ycs570.csv')
        X2=pd.DataFrame(X)
        prediction_rf=self.my_rf_algoritm(X)
        prediction_GradientBoostingRegressor=self.my_GradientBoostingRegressor_algoritm(X)
        prediction_ElasticNet=self.my_ElasticNet_algoritm(X)
        prediction_linearRegressor=self.my_linearRegressor_algoritm(X)
        prediction_XGBregresor=self.my_XGBregresor_algoritm(X)
        prediction_KNN=self.my_KNN_algoritm(X)


        X2['prediction_rf'] = prediction_rf
        X2['prediction_GradientBoostingRegressor'] = prediction_GradientBoostingRegressor
        X2['prediction_ElasticNet'] = prediction_ElasticNet
        X2['prediction_linearRegressor'] = prediction_linearRegressor
        X2['prediction_XGBregresor'] = prediction_XGBregresor
        X2['prediction_KNN'] = prediction_KNN


        return X2


#La situiente clase es para añadir una columna con los resultados de un clusttering.

class Columm_of_KMeans(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.my_KMeans_algoritm=None

    def fit(self, X, y):
        self.my_KMeans_algoritm=KMeans().fit(X).predict
        return self

    def transform(self, X):
        X2=pd.DataFrame(X)
        prediction_KMeans=self.my_KMeans_algoritm(X)
        X2['prediction_KMeans'] = prediction_KMeans
        return X2

#La siguiente clase la defino para eliminar la columna id
class Drop_columns(BaseEstimator, TransformerMixin):

    def __init__(self, columns=['id']):
        self.columns = columns


    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.drop(self.columns,1)

#la siguiente clase la defino para eliminar las etiquetas que no aparezcan en train y en test (no solo en uno de ellos), para una caracteristica.
#necesitaremos, ademas, la siguiente función (en realidad se puede hacer directamente, pero parece mas complicado si no queremos hacer bucles)
def filter_remove(x,remove):
    if x in remove:
        return np.nan
    return x

class Drop_labels(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns=columns

    def fit(self, X, y):
        return self

    def filter_remove(x):
        if x in self.remove:
            return np.nan
        return x

    def transform(self, X):
        config = get_config()
        data_2, processed_2 = config.DATA_READER.read()
        X_2, y_2, test_2 = split_data(data_2)

        for column in list(X.select_dtypes(include=['object']).columns):
            if not X_2[column].nunique() == test_2[column].nunique()==y_2[column].nunique():
                set_X_2 = set(X_2[column].unique())
                set_y_2 = set(y_2[column].unique())
                set_test_2 = set(test_2[column].unique())
                remove_X_2 = set_X_2 - (set_X_2.intersection(set_test_2)).intersection(set_y_2)
                remove_test_2 = set_test_2 - (set_X_2.intersection(set_test_2)).intersection(set_y_2)
                remove_y_2 = set_y_2 - (set_X_2.intersection(set_test_2)).intersection(set_y_2)
                remove = remove_X_2.union(remove_test_2).union(remove_y_2)

                X[column] = X[column].apply(lambda x: filter_cat(x,remove), 1)
        return X



class strings_a_floats(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns


    def fit(self, X, y):
        return self

    def transform(self, X):

        columns = self.columns
        if not columns:
            columns = X.columns

        for c in columns:
            X[c]=X[c].astype(np.float64)

        return X

class Mi_EliminaNaN(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns


    def fit(self, X, y):
        return self

    def transform(self, X):

        columns = self.columns
        if not columns:
            columns = X.columns

        for c in columns:
            if X[c].dtype==float:
                X[c]=X[c].fillna(X[c].mean())
            elif X[c].dtype==int:
                X[c]=X[c].astype(float)
                X[c] = X[c].fillna(X[c].mean())
            else:
                if len(X[c].unique())>20:
                    X[c] = X[c].fillna(X[c][0])
                else:
                    valores = pd.Series(X[c].unique()).dropna()
                    maximo = valores[0]
                    for maximo2 in valores:
                        if (X[c] == maximo).sum() < (X[c] == maximo2).sum():
                            maximo = maximo2
                    X[c] = X[c].fillna(maximo)

        return X

    def invert_transform(self, y):
        return y



class Mi_EliminaNaN2(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns


    def fit(self, X, y):
        return self

    def transform(self, X):

        columns = self.columns
        if not columns:
            columns = X.columns

        for c in columns:
            if X[c].dtype==float:
                X[c]=X[c].fillna(99999999999999999999999999)
            elif X[c].dtype==int:
                X[c]=X[c].astype(float)
                X[c] = X[c].fillna(np.inf)
            else:
                if len(X[c].unique())>20:
                    X[c] = X[c].fillna(X[c][0])
                else:
                    valores = pd.Series(X[c].unique()).dropna()
                    maximo = valores[0]
                    for maximo2 in valores:
                        if (X[c] == maximo).sum() < (X[c] == maximo2).sum():
                            maximo = maximo2
                    X[c] = X[c].fillna(maximo)

        return X

    def invert_transform(self, y):
        return y


class Mi_MataNaN(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns


    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.dropna()

    def invert_transform(self, y):
        return y


class Drop_Borrar(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns


    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[['x1','x2','x3']]

    def invert_transform(self, y):
        return y

class RemoveStringsTransform(BaseEstimator, TransformerMixin):

    def __init__(self, strings, columns=None):
        self.strings = strings
        self.columns = columns


    def fit(self, X, y):
        return self

    def transform(self, X):

        columns = self.columns
        if not columns:
            columns = X.columns

        for c in columns:
            for string in self.strings:
                if X[c].dtype == np.object and X[c].str.contains(string).any():
                    X[c] = X[c].str.replace(string, '')

        return X


class ImputerTransform(BaseEstimator, TransformerMixin):

    def __init__(self, missing_values='NaN', strategy=-999999):
        self.missing_values = missing_values
        self.strategy = strategy


    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if self.strategy in ['mean', 'median', 'most_frequent']:

            imp = Imputer(self.missing_values, self.strategy)
            X = imp.fit_transform(X)

        else:
            for c in X.columns:

                if X[c].dtype == np.object:
                    strategy = str(self.strategy)
                else:
                    strategy = int(self.strategy)

                if self.missing_values == "NaN":
                    X.loc[pd.isnull(X[c].values), c] = strategy
                else:
                    X.loc[X[c].isin([self.missing_values]), c] = strategy

        return X


class LabelTransform(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.01, columns=None):
        self.encoders = None
        self.threshold = threshold
        self.single_encoder = None
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        threshold = self.threshold
        if not isinstance(threshold, int):
            threshold = int(X.shape[0] * self.threshold)

        if len(X.shape) > 1:
            if not self.encoders:
                self.encoders = {}

            columns = X.columns
            if self.columns:
                columns = self.columns

            for c in columns:
                if len(X[c].unique()) < threshold:
                    if c in self.encoders:
                        encoder = self.encoders[c]
                    else:
                        encoder = LabelEncoder()
                        encoder.fit(X[c])
                        self.encoders[c] = encoder


                    X[c] = encoder.transform(X[c])

        else:
            if not self.single_encoder:
                encoder = LabelEncoder()
                encoder.fit(X)
                self.single_encoder = encoder

            X = self.single_encoder.transform(X)

        return X

    def invert_transform(self, y):
        return y

class TfIdfTransform(BaseEstimator, TransformerMixin):

    def __init__(self, columns, params={}):
        self.columns = columns
        self.encoders = {}
        self.params = params

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        for c in self.columns:

            if c in self.encoders:
                encoder = self.encoders[c]
            else:
                encoder = TfidfVectorizer(**self.params)
                encoder.fit(X[c])
                self.encoders[c] = encoder

            features = pd.DataFrame(encoder.transform(X[c]).todense(), index=X.index)
            del X[c]
            X = pd.concat((X, features), axis=1)

        return X


    def invert_transform(self, y):
        return y


class InteractionTransform(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, interactions=['mult']):
        self.columns = columns
        self.interactions = interactions

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):

        columns = self.columns
        if not columns:
            columns = X.columns

        for i,c1 in enumerate(columns):
            for c2 in columns[i+1:]:
                for interaction in self.interactions:
                    c = '%s_%s_%s' % (c1, c2, interaction)
                    if interaction == 'mult':
                        X[c] = X[c1] * X[c2]

                    if interaction == 'div':
                        X[c] = X[c1] / X[c2]
                    if interaction == 'sub':
                        X[c] = X[c1] - X[c2]

                    if interaction == 'add':
                        X[c] = X[c1] + X[c2]

        return X

class LogTransform(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):

        if isinstance(X, pd.DataFrame):

            columns = self.columns
            if not columns:
                columns = X.columns

            for c in columns:
                X[c] = np.log(X[c])
        else:
            X = np.log(X)

        return X

    def invert_transform(self, X):
        return np.exp(X)



