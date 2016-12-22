from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import StandardScaler
from feature_selection import TreeBased
import persistance
from feature_transform import LabelTransform, TfIdfTransform

# Directorio donde están los datos (train.csv, test.csv)
from models import XGBClassifierWrapper

DATA_DIR = 'sentiment/data'
DATA_READER = persistance.FileReader
DATA_COLUMN_SEP = '\t'

SAVE_MODELS = True
MODELS_DIR = 'sentiment/models'

PREDICTIONS_DIR = 'sentiment/submissions'
PREDICTIONS_PREPROCESS = None
PREDICTION_COLUMNS = ['PhraseId']
NEED_PROBA = False

TARGET = "Sentiment"
TARGET_TRANSFORM = None

SCORER = make_scorer(accuracy_score,
                     needs_proba=NEED_PROBA,
                     greater_is_better=False)

RANDOM_STATE = 2016
JOBS = 1

PREPROCESSING = [

    ('base', [
        ("tf-idf", TfIdfTransform(columns=['Phrase'],
                                  params={'max_features': 100,
                                         'stop_words': 'english',
                                         'norm': 'l1'})),
    ]),
]

MODELS = {
    # ('lr', LogisticRegression(fit_intercept=True, solver='sag')),
    ('random_forest', RandomForestClassifier(n_estimators=150)),

}

FOLDS = {
    'generator': StratifiedKFold,
    'params': {
        'n_folds': 5,
        'shuffle': True,
        'random_state': RANDOM_STATE
    }
}

META_PARAMETER_OPTIMIZATION = False
META_PARAMETER_OPTIMIZER = GridSearchCV
MAX_PARAMETER_SEARCH_ITERATIONS = 100
META_PARAMETERS = {
    """
    'lr': {
        'fit_intercept': [True, False],
        'C': [200, 2000, 20000],
        'tol': [0.006]
    },

    'random_forest': {
        'n_estimators': [10],
        'criterion': ['gini'],
        'max_features': ['auto'],
        'max_depth': [None, 6, 12],
        'min_samples_split': [2],
        'min_samples_leaf': [5, 3, 1],
    },


    'KNN': {
        'n_neighbors': [5, 10, 20],
        'algorithm': ['ball_tree', 'kd_tree'],
        'leaf_size': [20, 30],
        'p': [1, 2]
    }
    """
}
