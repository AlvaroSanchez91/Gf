from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import median_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from xgboost import XGBRegressor

from feature_generation import XORFeatureGeneration
from feature_selection import TreeBased
import persistance
import os

from feature_transform import LogTransform , ImputerTransform, ModelStakingLevel1, ModelStakingLevel1_regresor, LabelTransform, Drop_labels, filter_remove, Columm_of_KMeans

DATA_DIR = 'allstate/data'
DATA_READER = persistance.FileReader
DATA_COLUMN_SEP = ','

SAVE_MODELS = True
MODELS_DIR = 'allstate/models'

PREDICTIONS_DIR = 'allstate/submissions'
PREDICTIONS_PREPROCESS = None
PREDICTION_COLUMNS = ['id']
NEED_PROBA = False

TARGET = "loss"
TARGET_TRANSFORM = LogTransform
SCORER = make_scorer(median_absolute_error, greater_is_better=False)

RANDOM_STATE = 2016
JOBS = 2

FEATURE_SELECTION_N = 20

PREPROCESSING = [

    ('select20_modelstaking', [
            ('Columm_of_KMeans',Columm_of_KMeans())
            #('Drop_labels',Drop_labels()),
            #('eliminoNaN',ImputerTransform()),
            #('LabelTransform', LabelTransform(columns=['cat%d' % i for i in range(1, 117)])),
            #('fs', TreeBased('extra_trees_regressor', 20, 20)),
            #('inter', InteractionTransform(
                #interactions=['sum'],
                #columns=['cat%d' % i for i in range(20, 30)])),
            #('eliminoNaN',ImputerTransform()),
            #('ModelStakingLevel1_regresor', ModelStakingLevel1_regresor()),
    ] ),

]
MODELS = {
    #('lr', LinearRegression(fit_intercept=True)),
    ('xgb', XGBRegressor()),

    #('lasso', Lasso(fit_intercept=True)),
    #('ridge', Ridge(fit_intercept=True)),
    #('elastic_net', ElasticNet(fit_intercept=True)),
    #('bayes_ridge', BayesianRidge(fit_intercept=True)),
    #('SGD', SGDRegressor()),
    #('random_forest', RandomForestRegressor()),
    #('gb', GradientBoostingRegressor()),
    #('KNN', KNeighborsRegressor(n_neighbors=10)),

}




FOLDS = {
    'generator': KFold,
    'params': {
         'n_folds': 3,
         'shuffle': True,
         'random_state': RANDOM_STATE
    }
}

META_PARAMETER_OPTIMIZATION = True
META_PARAMETER_OPTIMIZER = RandomizedSearchCV
MAX_PARAMETER_SEARCH_ITERATIONS = 100
META_PARAMETERS = {
    'lr': {
        'fit_intercept': [False],
        'normalize': [False],
    },

    'random_forest': {
        'n_estimators': [1],
        'criterion': ['mse'],
        'max_features': ['auto'],
        'max_depth': [6],
        'min_samples_split': [0.6],
        'min_samples_leaf': [5],
    },

    'gb': {
        'n_estimators': [1],
        'max_features': ['auto'],
        'max_depth': [6],
        'min_samples_split': [0.6],
        'min_samples_leaf': [5],
    },

    'xgb': {
        #'n_estimators': [10 , 20],
        'min_child_weight':[3],
        #'eta': [0.01],
        'colsample_bytree': [0.6],
        'max_depth': [12],
        'subsample': [0.8],
        'reg_alpha': [1],
        'gamma': [1],
        'silent': [1],
        #'verbose_eval': [True],
        #'seed': RANDOM_STATE
    },



    'KNN': {
        'n_neighbors': [5],
    },
    'elastic_net':{
        'alpha':[0,0.3,0.7,1],

    }
}




