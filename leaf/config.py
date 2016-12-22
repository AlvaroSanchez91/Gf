from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from feature_selection import TreeBased
import persistance
from feature_transform import ModelStakingLevel1 ,LabelTransform, InteractionTransform

DATA_DIR = 'leaf/data'
DATA_READER = persistance.FileReader
DATA_COLUMN_SEP = ','

SAVE_MODELS = False
MODELS_DIR = 'leaf/models'

PREDICTIONS_DIR = 'leaf/submissions'
PREDICTIONS_PREPROCESS = None
PREDICTION_COLUMNS = ['id']
NEED_PROBA = True

TARGET = "species"
TARGET_TRANSFORM = LabelTransform

SCORER = make_scorer(log_loss,
                     needs_proba=True,
                     greater_is_better=False)

RANDOM_STATE = 2016
JOBS = 1

FEATURE_SELECTION_N = 20

PREPROCESSING = [

    #('none', [] ),

    #('select', [
    #    ('fs',  TreeBased('extra_trees_regressor', 20, 20)),
    #]),


    ('scaler_int', [
        ('inter', InteractionTransform(
            interactions=['add'],
            columns=['margin%d' % i for i in range(1, 10)])),
        ('std', StandardScaler()),
        ('fs', TreeBased('extra_trees_regressor', 20, 190)),
        ('ModelStakingLevel1', ModelStakingLevel1()),

    ]),

    ('scaler', [
        ('std', StandardScaler()),

    ]),

    ('scaler_fs', [
        ('std', StandardScaler()),
        ('fs', TreeBased('extra_trees_regressor', 20, 20)),
    ]),

]

MODELS = {
    #('lr', LogisticRegression(fit_intercept=True, solver='newton-cg',
    #                          multi_class='multinomial')),
    ('rf', RandomForestClassifier(random_state=RANDOM_STATE)),
    #('xgb', XGBRegressor()),

}

FOLDS = {
    'generator': StratifiedKFold,
    'params': {
         'n_folds': 3,
         'shuffle': True,
         'random_state': RANDOM_STATE,
    }
}

META_PARAMETER_OPTIMIZATION = True
META_PARAMETER_OPTIMIZER = GridSearchCV
MAX_PARAMETER_SEARCH_ITERATIONS = 100
META_PARAMETERS = {

    'lr': {
        'fit_intercept': [True, False],
        'C': [200, 2000, 20000],
        'tol': [0.006]
    },


    'rf': {
        'n_estimators': [20],
        'criterion': ['gini'],
        'max_features': ['auto'],
        'max_depth': [None],
        'min_samples_split': [1],
        'min_samples_leaf': [3, 1]
    },


    'KNN': {
        'n_neighbors': [5, 10, 20],
        'algorithm': ['ball_tree', 'kd_tree'],
        'leaf_size': [20, 30],
        'p': [1, 2]
    }
}




