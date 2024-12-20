NDVI_FIRST_COLUMN = 22
ONLY_USE_SUBSET = False
HOLDOUT_MODE = False
USE_AUTOML = False
USE_HOLDOUT_SCALING = False
TRAIN_DATA = "pre-processed-final_all_index_231024.xlsx - MTCI"
TEST_DATA = "" 
COLUMNS_TO_REMOVE = ['Date', 'Internal code', 'Location', 'Depth', 'Texture', 'Active CaCO3 (%)',
                      'ng/μl', 'Α260/280']
TRAIN_YEARS = "2021"
MODE = "window"
OUTPUT_COLUMN_NAME = "Yield"
USE_POLYNOMIAL = True
# USE_SPLINES = True
POLYNOMIAL_DEGREE = 1
SMOOTHING_ORDER = 2
SAMPLING_MODE = "constant"
SEED = 2024
NUM_FOLDS = 5
FORCED_COLUMN_IX = None