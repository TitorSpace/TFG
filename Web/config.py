import os

# Define the root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths for datasets
DATASETS_DIR = './Web/datasets'
DATASET_TESTEO_MODELO_CSV = os.path.join(DATASETS_DIR, 'dataset_testeo_modelo.csv')
OPENFOODFACTS_CSV = os.path.join(DATASETS_DIR, 'openFoodFactsProducts.csv')
LISTA_TOTAL_INGREDIENTES_CSV = os.path.join(DATASETS_DIR, 'listaTotalIngredientes.csv')
LISTA_TOP_FREQ_CSV = os.path.join(DATASETS_DIR, 'listaTopFreq.csv')
DATASET_REFERENCIA_PRODUCTOS_CSV = os.path.join(DATASETS_DIR, 'dataset_referencia_productos.csv')
DATASET_TESTEO_MODELO_OUTPUT_CSV = os.path.join(DATASETS_DIR, 'Data_mining', 'dataset_testeo_modelo.csv')

# Define paths for models
ML_MODEL_DIR = './Web/ml_model'
MODELO_GRANDE_NUTRISCORE_H5 = os.path.join(ML_MODEL_DIR, 'modelo_grande_nutriscore.h5')
MODELO_GRANDE_NOVASCORE_H5 = os.path.join(ML_MODEL_DIR, 'modelo_grande_novascore.h5')



# Path for feedback CSV
FEEDBACK_CSV_PATH = os.path.join(DATASETS_DIR, 'feedback.csv')

# Path for product list
LISTA_PRODUCTOS_CSV = os.path.join(DATASETS_DIR, 'listaProductos.csv')