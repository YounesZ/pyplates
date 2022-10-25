from os import path

# --- GENERAL
DEVICE = 'cpu'
RANDOM_SEED = 42
DF_DATA_DECIMAL = ','
DF_LABEL_DECIMAL = '.'
DF_DATA_SEPARATOR = ';'
DF_LABEL_SEPARATOR = ','
REQUIRED_LOCAL_VARIABLES = ['PATH_TO_CODE', 'PATH_TO_DATA', 'PATH_TO_MODELS']

# --- HYPERPARAMETERS
HYPERPARAM_RANGE_TYPE = ['continuous', 'discrete']

# --- DATASETS
LOW_CLUTCH_NAME = 'clutch_low_slip'
HIREV_CLUTCH_NAME = 'clutch_hi_slip'

# --- DATA PRE-PROCESSING
MINMAXSCALER_RANGE = (0, 1)
N_SHIFTS_RECURRENT_MODELLING = 5

# --- FEATURE ENGINEERING
PRETRAINED_ENCODERS = {'clutch_slip': 'Nov17_18-57-51_compute01',
                       'cvt_ratio_actual': 'Nov17_23-27-58_compute01',
                       'cvt_speed_in': 'Nov18_03-58-55_compute01',
                       'cvt_speed_out': 'Nov18_08-33-02_compute01',
                       'engine_torque': 'Nov18_13-04-44_compute01'}
FEATURES_TIME_SERIES = ['clutch_slip', 'cvt_ratio_actual', 'cvt_speed_in', 'cvt_speed_out', 'engine_torque']
FEATURES_SCALAR = ['fuel_pedal_pos', 'temp_sump']
TARGET_COLUMN = 'kissPoint'

# --- MODELLING
TRAINING_EPOCHS = 1

# --- TRACKING
LOCAL_MODEL_STORE = '/Users/younes/Documents/Perso/Models/pipeline_backups'
MLFLOW_TRACKER_URI = 'localhost:5005'
MLFLOW_EXPERIMENT_NAME = 'mlcars'
MLFLOW_EXCLUDE_PARAMS = ['X_TRAIN', 'X_TEST', 'Y_TRAIN', 'Y_TEST', 'FEATURES_COLUMNS', 'all_features', 'FEATURES_ENCODING_DICT']


def read_default_config():
    params = {'FEATURES_TIME_SERIES': FEATURES_TIME_SERIES,
              'FEATURES_SCALAR': FEATURES_SCALAR,
              'TARGET_COLUMN': TARGET_COLUMN,
              'MINMAXRANGE': MINMAXSCALER_RANGE,
              'N_SHIFTS_RECURRENT_MODELLING': N_SHIFTS_RECURRENT_MODELLING,
              'DEVICE': DEVICE,
              'TRAINING_EPOCHS': TRAINING_EPOCHS}
    return params

