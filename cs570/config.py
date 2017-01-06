from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import median_absolute_error
from sklearn.metrics import auc


from feature_selection import TreeBased
import persistance
from feature_transform import ImputerTransform, LogTransform ,ModelStackingLevel1 ,save_processed_data ,DummyId,LabelTransform, InteractionTransform, RemoveStringsTransform,strings_to_floats,My_replaceNaN1, My_replaceNaN2, My_DropNan

DATA_DIR = 'cs570/data'
DATA_READER = persistance.FileReader
DATA_COLUMN_SEP = ','

SAVE_MODELS = True
MODELS_DIR = 'cs570/models'

PREDICTIONS_DIR = 'cs570/submissions'
PREDICTIONS_PREPROCESS = [ DummyId() ]
PREDICTION_COLUMNS = ['id']
NEED_PROBA = False

TARGET = "y"
TARGET_TRANSFORM = None


#SCORER = make_scorer(auc, greater_is_better=True)
SCORER = make_scorer(median_absolute_error, greater_is_better=False)
#SCORER = make_scorer(log_loss,
 #                    needs_proba=True,
  #                   greater_is_better=False)

RANDOM_STATE = 2016
JOBS = 2

FEATURE_SELECTION_N = 20

PREPROCESSING = [
    ('My_preprocessing', [
        #('My_DropNan',My_DropNan()),
        ('rep',RemoveStringsTransform(strings=['$','%'],columns=['x32','x37'])),
        ('qutastrings',strings_to_floats(columns=['x32','x37'])),
        ('eliminoNaN',ImputerTransform()),
        ('LabelTransform',LabelTransform( columns=['x24','x29','x30'])),
        #('inter', InteractionTransform(
         #   interactions=['sum'],
          #  columns=['x%d' % i for i in range(20, 30)])),
        #('std', StandardScaler()),
        #('fs', TreeBased('extra_trees_regressor', 20, 20)),
        #('GuardaDatosProcesados', GuardaDatosProcesados()),
        #('ModelStackingLevel1', ModelStackingLevel1()),
        #('GuardaDatosProcesados2', save_processed_data(nombre="cs570con20colum2.csv")),
    ])

]


MODELS = {
    ('lr', LogisticRegression(fit_intercept=True, solver='newton-cg',
                              multi_class='multinomial')),
    #('rf', RandomForestClassifier(random_state=RANDOM_STATE)),
    #('xgb', XGBRegressor()),
    #('xgbc', XGBClassifier()),

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
META_PARAMETER_OPTIMIZER = RandomizedSearchCV
MAX_PARAMETER_SEARCH_ITERATIONS = 100
META_PARAMETERS = {

    'lr': {
        'fit_intercept': [True, False],
        'C': [200, 2000, 20000],
        'tol': [0.006]
    },


    'rf': {
        'n_estimators': [100],
        'criterion': ['gini'],
        'max_features': ['auto', 'sqrt','log2'],
        'max_depth': [5],
        'min_samples_split': [1],
        'min_samples_leaf': [1]
    },


    'KNN': {
        'n_neighbors': [5, 10, 20],
        'algorithm': ['ball_tree', 'kd_tree'],
        'leaf_size': [20, 30],
        'p': [1, 2]
    },
    'xgb': {
        'n_estimators': [20],
        'max_depth':[25],
        'min_child_weight':[1,3],
        'gamma':[0,1],
        'subsample':[0.5,1],
        'colsample_bytree':[0.5,1],
        'reg_lambda':[0.01,1],
        'reg_alpha':[0,1]
    },
    'xgbc': {
        'n_estimators': [20],
        'max_depth':[25],
        'min_child_weight':[1,3],
        'gamma':[0,1],
        'subsample':[0.5,1],
        'colsample_bytree':[0.5,1],
        'reg_lambda':[0.01,1],
        'reg_alpha':[0,1]
    },
}



