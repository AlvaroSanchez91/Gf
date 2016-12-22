from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
import persistance

# En RANDOM_STATE indicamos la semilla usada para la generación de número aleatorios
RANDOM_STATE = 2016

# En JOBS indicamos cuando procesos en paralelo queremos usar
JOBS = 1

###
### LECTURA DE DATOS
###

# En DATA_DIR indicamos donde están los datos (train.csv y test.csv)
DATA_DIR = '[directorio del projecto]/data'

# Esta es la case que se escargará de hacer la lectura de los datos. No modificar
DATA_READER = persistance.FileReader

# Este es el separador de columnas en el archivo CSV. Normalmente es una coma, pero
# también podría ser otros como un tabulador('/t') o un barra vertical (|)
DATA_COLUMN_SEP = ','

###
### GUARDADO DE MODELOS
###

# Con SAVE_MODELS indicamos que queremos guardar los modelos que nos genere
# el framework. Al principio, cuando estamos probando los parámetros iniciales
# no será neceario guardar los modelos. Cuando hayamos realizado una primera
# afinación de parámetros le pasemos un grid search para buscar mejores parámetros
# sería recomendable ir guardando los modelos que nos genere
SAVE_MODELS = False

# Con MODELS_DIR indicamos el directorio donde guardará los modelos
MODELS_DIR = '[directorio del projecto]/models'

###
### PREDICCIONES
###

# PREDICTIONS_DIR indica el directorio donde se guardarán las predicciones
PREDICTIONS_DIR = '[directorio del projecto]/submissions'

# En ocasiones antes de crear el archivo de predicciones puede ser necesario
# realizar un preprocesado. Por ejemplo, si no nos porporcionan una columna 'id'
# pero es necesaria en el archivo de predicciones, antes tendremos que generarla
PREDICTIONS_PREPROCESS = None

# Estas son las columnas adicionales a las predicciones que quereremos añadir al
# archivos de predicciones. Normalmente solamente el id, pero podrían ser más...
PREDICTION_COLUMNS = ['id']

# TARGET es el nombre de la columna que queremos predecir o clasificar.
TARGET = "[Nombre de la columnas respuesta (y)]"

# En TARGET_TRANSFORM podemos indicar una clase que se encarga de transformar
# la columna respuesta. Por ejemplo, si estamos en un problema de clasificación y
# las categorías nos vienen en modo texto, primero tendremos que convertilas a
# numérico.
TARGET_TRANSFORM = None

# En lo problemas de clasificación nos puede pedir la probabilidad de que cada
# muestra pertenezca a cada clase. Para ello pondremos NEED_PROBA = True
NEED_PROBA = False


#
# PREPROCESADO
#

# En PREPROCESSING podemos indicar una lista de transformaciones que queremo probar.
# Cada elemento de la lista, es un conjunto de transformaciones que se ejecutarán
# una tras otra usando la clase PipeLine de Scikit Learn.
PREPROCESSING = [
    '''
    ('nombre del conjunto de transformaciones 1', [
        ("transformacion 1.1", Transformacion11()),
        ("transformacion 1.2", Transformacion12()),
    ]),

    ('nombre del conjunto de transformaciones 2', [
        ("transformacion 2.1", Transformacion21()),
    ]),
    '''

]

#
# MODELOS
#

# En MODELS podemos indicar una lista de modelos que queremos entrenar. Cada
# elemento esta compuesto por un nombre y una instancia de un algoritmo

MODELS = {
    '''
    ('nombre modelo 1', Algoritmo1(parametros....)),
    ('nombre modelo 2', Algoritmo2(parametros....)),
    '''
}

#
# VALIDACIÓN CRUZADA
#

# En el parámetro FOLDS indicamos para crear los folds queremos aplicar.
# Tenemos en Scikit learn varias clases para crear los folds. Las más
# habituales son: KFold y StratifiedKFold.

FOLDS = {
    'generator': StratifiedKFold,
    'params': {
         'n_folds': 5,
         'shuffle': True,
         'random_state': RANDOM_STATE
    }
}

#
# OPTIMIZACIÓN DE PARÁMETROS
#

# Con META_PARAMETER_OPTIMIZATION indicamos si queremos usar optimización de parámetros.
# Si este parámetro esta a False se usarán los pametros indicados en MODELS
META_PARAMETER_OPTIMIZATION = True

# En META_PARAMETER_OPTIMIZER indicamos el algoritmo a usar para optimizar los parámetros.
# Los más habituales son GridSearchCV y RandomSearchCV
META_PARAMETER_OPTIMIZER = GridSearchCV

# Si META_PARAMETER_OPTIMIZER=RandomSearchCV este parámetro indica cuantos modelos
# con parámetros aleatorios se probarán
MAX_PARAMETER_SEARCH_ITERATIONS = 100

# En META_PARAMETERS indicamos los conjunto de parametros que queremos probar.
# GridSearchCV probará todas las combinaciones posibles y RandomSearchCV seleccionará
# aleatoriamente conjuntos de parámetros de dichas combinaciones.
META_PARAMETERS = {
    """
    'nombre modelo 1': {
        'parametro 1': [True, False],
        'parametro 2': [200, 2000, 20000],
        ...
        ...
    },

    'nombre modelo 2': {
        'parametro 1': [10],
        'parametro 2': ['gini'],
        ...
        ...
    },
    """
}



