from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import median_absolute_error

from feature_selection import TreeBased
import persistance
from feature_transform import LogTransform ,ModelStakingLevel1 ,GuardaDatosProcesados ,DummyId,LabelTransform, InteractionTransform, RemoveStringsTransform,strings_a_floats,Mi_EliminaNaN, Mi_EliminaNaN2

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

SCORER = make_scorer(median_absolute_error, greater_is_better=False)
#SCORER = make_scorer(log_loss,
 #                    needs_proba=True,
  #                   greater_is_better=False)

RANDOM_STATE = 2016
JOBS = 2

FEATURE_SELECTION_N = 20

PREPROCESSING = [

    #('none', [] ),

    #('select', [
    #    ('fs',  TreeBased('extra_trees_regressor', 20, 20)),
    #]),


    ('scaler_int', [
        #('Drop_Borrar',Drop_Borrar()),
        #('Mi_MataNaN',Mi_MataNaN()),
        #('Elimina_Filas',Elimina_Filas()),
        ('rep',RemoveStringsTransform(strings=['$','%'],columns=['x32','x37'])),
        ('qutastrings',strings_a_floats(columns=['x32','x37'])),
        ('eliminoNaN',Mi_EliminaNaN()),

        ('LabelTransform',LabelTransform( columns=['x24','x29','x30'])),
        #('inter', InteractionTransform(
         #   interactions=['sum'],
          #  columns=['x%d' % i for i in range(20, 30)])),
        #('std', StandardScaler()),
        #('fs', TreeBased('extra_trees_regressor', 20, 20)),
        #('GuardaDatosProcesados', GuardaDatosProcesados()),
        #('ModelStakingLevel1', ModelStakingLevel1()),
        #('GuardaDatosProcesados2', GuardaDatosProcesados(nombre="cs570con20colum2.csv")),
    ])

    #('scaler', [
     #   ('std', StandardScaler()),

    #]),

    #('scaler_fs', [
     #   ('std', StandardScaler()),
      #  ('fs', TreeBased('extra_trees_regressor', 20, 20)),
    #]),

]


MODELS = {
    #('lr', LogisticRegression(fit_intercept=True, solver='newton-cg',
     #                         multi_class='multinomial')),
    #('rf', RandomForestClassifier(random_state=RANDOM_STATE)),
    ('xgb', XGBRegressor()),

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
        'n_estimators': [20,100],
        'max_depth':[25],
        'min_child_weight':[1],
        'gamma':[0,0.05,0.1,0.3,0.5,0.7,0.9,1],
        'subsample':[0.5,0.6,0.7,0.8,0.9,1],
        'colsample_bytree':[0.5,0.6,0.7,0.8,0.9,1],
        'reg_lambda':[0.01,0.05,0.1,0.3,0.5,0.7,0.9,1],
        'reg_alpha':[0,0.1,0.5,1]
    }
}



