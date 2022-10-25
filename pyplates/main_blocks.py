import pickle
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as  plt

from os import path, makedirs, environ, listdir
from typing import List
from mlflow import set_tracking_uri, \
                   start_run, \
                   log_metric, \
                   log_param
from inspect import getmembers, isroutine, getargspec
from sklearn.base import TransformerMixin, RegressorMixin
from mlflow.pyfunc import log_model, PythonModel
from pyplates.utils import get_mlflow_experiment_id, mlflow_log_params
from sklearn.metrics import mean_absolute_error
from pyplates.config.general import HYPERPARAM_RANGE_TYPE, RANDOM_SEED, DF_DATA_SEPARATOR, \
                                    DF_DATA_DECIMAL, DF_LABEL_SEPARATOR, DF_LABEL_DECIMAL, \
                                    MLFLOW_TRACKER_URI, MLFLOW_EXPERIMENT_NAME, LOCAL_MODEL_STORE,\
                                    REQUIRED_LOCAL_VARIABLES
from pytox.utils.data_loaders import load_datafile
from pyplates.config.localvars import PATH_TO_CODE
from pytox.utils.data_structures import check_input, convert_input, serialize_dictionary
from pytox.utils.signal_processing import ts_custom_split, get_sample_variables

import logging

logging.getLogger("mlflow").setLevel(logging.DEBUG)


set_tracking_uri(MLFLOW_TRACKER_URI)
environ["MLFLOW_TRACKING_USERNAME"] = "youbuntu"
environ["MLFLOW_TRACKING_PASSWORD"] = "Dr!ss0un0urs"
MLFLOW_EXPERIMENT_ID = get_mlflow_experiment_id(MLFLOW_EXPERIMENT_NAME)

"""
AWS_ACCESS_KEY_ID = "AKIAVJZK6MGAMX4KBJV5"
AWS_SECRET_ACCESS_KEY = "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"
"""


class Block(object):

    def __init__(self, hyper_range={}, **kwargs):
        # TODO: FIX HYPERPARAMETER RANGE DEFINITION
        #self.define_hyperparameter_range(hyper_range, **kwargs)
        #self.check_hyperparameter_range()
        return

    def hyperparameter_range(self):
        raise NotImplementedError('A range for each hyperparameter must be defined for a block')

    def name(self):
        raise NotImplementedError('A name must be defined for the block')

    def define_hyperparameter_range(self, default_range, manual_range):
        # Get list of all hyperparams
        attributes = getmembers(self, lambda a: not (isroutine(a)))
        hyperparams = dict([a for a in attributes
                            if not (a[0].startswith('__') and a[0].endswith('__'))])

        # Loop on hyperparameters default values
        for i_h in default_range.keys():
            if "default" not in default_range[i_h]:
                raise NotImplementedError("The default value for hyperparameter %s not defined" % i_h)
            # Range value
            if "range" not in default_range[i_h].keys():
                rvalue = [default_range[i_h]["default"]]
            elif (not isinstance(rvalue, list)) or (len(range_value) < 1) or (len(range_value) > 2):
                raise ValueError('Hyperparameter range value must be either 1d or 2D')
            else:
                rvalue = default_range[i_h]["range"]

            # Check range type
            if "range_type" not in default_range[i_h]:
                rtype = detect_vec_type(rvalue)
            else:
                rtype = default_range[i_h]["range_type"]
            if rtype not in HYPERPARAM_RANGE_TYPE:
                raise ValueError('Hyperparameter range type must be within %s', HYPERPARAM_RANGE_TYPE)

            # Check values
            rdefault = default_range[i_h]["default"]
            # TODO: make sure default is within range and same type

            # append to defined ranges
            setattr(self, i_h, rdefault)
            hyperparams.pop(hyperparams.index(i_h))
            self.hyperparameter_range[i_h] = {'type': range_type,
                                              'value': range_value}

        # check for missing hyperparameters
        if len(hyperparams) > 0:
            raise ValueError('Range was not defined for hyperparameters: %s' % hyperparams)
        return

    def check_hyperparameter_range(self):
        # Get the list of class attributes
        attributes = getmembers(self, lambda a: not (isroutine(a)))
        hyperparams= dict([a for a in attributes
                           if not (a[0].startswith('__') and a[0].endswith('__'))])

        # Get the list of hyperparamter ranges
        hyperranges= self.hyperparameter_range

        # --- Tests
        # Make sure every parameter has a defined range (exists in dictionary)
        for i_ in hyperparams.keys():
            if i_ not in hyperranges.keys():
                setattr(self, 'range_' + i_, [hyperparams[i_]])


class  TransformerBlock(TransformerMixin, Block):

    def __init__(self, hyper_range={}):
        super().__init__(hyper_range)

    def fit_transform(self, X, y=None, **fit_params):
        raise NotImplementedError('A fit_tansform method must be implemented for transformers')

    def transform(self, X, y=None, **kwargs):
        raise NotImplementedError('A tansform method must be implemented for transformers')

    def save(self, save_path, **kwargs):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, save_path, **kwargs):
        with open(save_path, 'rb') as f:
            self = pickle.load(f)
        return self


class PostprocessorBlock(TransformerMixin, Block):

    def __init__(self, comment='', **kwargs):
        self.is_fit = False
        self.comment = comment
        super().__init__()

    def fit_transform(self, y, y_true=None, **fit_params):
        raise NotImplementedError('A fit_tansform method must be implemented for postprocessors')

    def transform(self, y, y_true=None, **kwargs):
        raise NotImplementedError('A tansform method must be implemented for postprocessors')

    def save(self, save_path, **kwargs):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, save_path, **kwargs):
        with open(save_path, 'rb') as f:
            self = pickle.load(f)
        return self


class ModelBlock(Block):

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, **fit_params):
        # Check model type
        mdl_type = self.get_model_type()
        if mdl_type == 'sklearn':  # Only accepts np.ndarrays as inputs
            X = np.concatenate(X, axis=0)
            y = np.concatenate(y, axis=0)
        elif mdl_type == 'toolbox':
            pass
        else:
            raise TypeError('Unrecognized model type : %s' % mdl_type)
        fit_result = self.model.fit(X, y)
        return fit_params

    def predict(self, X, y=None, **kwargs):
        # check input format - must be either array or list of arrays
        X_, y_ = check_input(X, y, x_type=List[np.ndarray])
        truth, pred = [], []
        for i_,j_ in zip(X_, y_):
            # Determine call type
            fcnsign = getargspec(self.model.predict)
            if fcnsign.keywords == 'kwargs':
                prediction = self.model.predict(i_, j_, **kwargs)
            elif fcnsign.keywords is None:
                prediction = self.model.predict(i_)

            if isinstance(prediction, tuple) and (len(prediction)==2):
                truth += [prediction[0]]
                pred += [prediction[1]]
            else:
                truth +=[j_]
                pred += [prediction]
        # Ensure ground truth and prediction are in the right format
        #assert isinstance(y, np.ndarray)
        #assert isinstance(pred, np.ndarray)
        #assert np.allclose(pred.shape, y.shape)
        return truth, pred, kwargs

    def _fit(self, X, y, **fit_params):
        # --- Basic training routine
        # Train model
        clf = self.model.fit(X, y)
        y_ = clf.predict(X)

        # Make output
        fit_results = {'train_score': mean_absolute_error(y_, y)}
        return fit_results

    def get_model_type(self):
        # Get model module tree
        module_tree = self.model.__module__.split('.')
        sklearn= module_tree[0]=='sklearn'
        toolbox= 'PythonToolBox' in module_tree
        if sklearn and toolbox:
            raise TypeError('Model type detection error - both sklearn and cvtcorp')
        elif not(sklearn or toolbox):
            raise TypeError('Model type detection error - not sklearn nor toolbox')
        elif sklearn:
            return 'sklearn'
        elif toolbox:
            return 'toolbox'

    def save(self, full_path, **kwargs):
        # Check model type
        mdl_type = self.get_model_type()
        mdl_path = '%s_%s' % (full_path, mdl_type)
        if mdl_type == 'sklearn':  # Only accepts np.ndarrays as inputs
            with open(mdl_path+'.pkl', 'wb') as f:
                pickle.dump([self.comment, self.model], f)
        elif mdl_type == 'toolbox':
            # Make folder
            # --- Fill folder
            os.makedirs(mdl_path, exist_ok=True)
            # Save model variables
            var_path = path.join(mdl_path, 'model_variables.pkl')
            input_shp= None
            if hasattr(self.model, 'input_shape'):
                input_shp = self.model.input_shape
            with open(var_path, 'wb') as f:
                pickle.dump([self.comment, self.model.__class__, input_shp], f)
            # Save model itself
            self.model.save(mdl_path)
        else:
            raise TypeError('Unrecognized model type : %s' % mdl_type)

    def load(self, full_path, **kwargs):
        # Check model type
        mdl_type = full_path.split('_')[-1].split('.')[0]
        if mdl_type == 'sklearn':  # Only accepts np.ndarrays as inputs
            with open(full_path, 'rb') as f:
                self.comment, self.model = pickle.load(f)
        elif mdl_type == 'toolbox':
            # Load model variables
            var_path = path.join(full_path, 'model_variables.pkl')
            with open(var_path, 'rb') as f:
                self.comment, mdl_class, input_shp = pickle.load(f)
            self.model = mdl_class(input_shape=input_shp).load(full_path)
        else:
            raise TypeError('Unrecognized model type : %s' % mdl_type)
        return self


class PipelineBlock(RegressorMixin, Block):

    def __init__(self, steps, verbose=1, load_mode=False, **params):
        self.verbose = verbose
        super().__init__()
        # Init pipeline blocks
        if not load_mode:
            steps_ = steps[1:-1]
            steps_ += [XYBlock(comment='XY splitter')]
            steps_ += [steps[-1]]
            self.steps = steps_
            # Prepare dataset
            self.X, self.y, _ = steps[0].fit_transform(**params)
            self.is_fit = False
            self.run_tests()
            self.pipeline_type = 'standard_v0'
        else:
            self.steps = []

    def run_tests(self):
        # --- 1. LOCAL VARIABLES
        # Make sure the local variables file exists
        PATH_TO_LOCAL = path.join(PATH_TO_CODE, 'config', 'localvars.py')
        if not path.isfile(PATH_TO_LOCAL):
            raise FileNotFoundError('The file localvars.py was not found at the root. '
                                    'Check out the readme for the format')
        # Check the local variables values
        from config import localvars
        localvars = localvars.__dict__
        # Loop on required local variables
        for i_ in REQUIRED_LOCAL_VARIABLES:
            # Make sure they're defined
            if i_ not in localvars:
                raise ValueError('The local variable %s is not defined in localvars.py' % i_)
            # Make sure the folder exists
            if not path.isdir(localvars[i_]):
                raise ValueError('Make sure the folder defined for %s is valid' % i_)

    def train(self, **fit_params):
        # Fit the pipeline
        environ["MLFLOW_TRACKING_USERNAME"] = "youbuntu"
        environ["MLFLOW_TRACKING_PASSWORD"] = "Dr!ss0un0urs"
        with start_run(experiment_id=MLFLOW_EXPERIMENT_ID,
                       run_name='pipeline_training') as run:
            fit_params = self.fit(X=self.X, y=self.y, **fit_params)
            mlflow_log_params(fit_params)
            self.save(**fit_params)

            # Train score
            y_, pred, _ = self.predict(self.X, self.y, **fit_params)
            score_train = self.score(y_, pred)
            log_metric("train_mae", score_train)

            # Test score
            y_, pred, _ = self.predict(fit_params['X_TEST'], fit_params['Y_TEST'], **fit_params)
            score_test = self.score(y_, pred)
            log_metric("test_mae", score_test)
        return score_train, score_test, fit_params

    def fit(self, X, y=None, **fit_params):
        for i_block in self.steps[:-1]:
            print('Fitting block %s' % i_block.__class__.__name__)
            X, y, fit_params = i_block.fit_transform(X, y, **fit_params)
        # Instantiate model
        fit_params['input_shape'] = check_input(X, x_type=List[np.ndarray])[0][0].shape[1:]
        try:
            self.steps[-1].model = self.steps[-1].model(**fit_params)
        except:
            try:
                self.steps[-1].model = self.steps[-1].model()
            except:
                raise TypeError('Unrecognized function signature')
        self.steps[-1].fit(X, y, **fit_params)
        self.is_fit = True
        return fit_params

    def predict(self, X, y=None, **params):
        for i_block in self.steps[:-1]:
            X, y, params = i_block.transform(X, y, **params)
        # Model prediction
        y_, pred, params = self.steps[-1].predict(X, y, **params)
        return y_, pred, params

    def score(self, y, y_, **fit_params):
        # check input formats
        y, _ = check_input(y, x_type=List[np.ndarray])
        y_, _ = check_input(y_, x_type=List[np.ndarray])
        # Call the model's scoring function
        scr = np.mean( [mean_absolute_error(i_, j_) for i_,j_ in zip(y, y_)] )
        return scr

    def infer(self, X, y=None, **kwargs):
        return self.predict(X, y, **kwargs)

    def performance_summary(self, split='train', **kwargs):
        # Make sure pipeline was fit
        assert self.is_fit
        # Make predictions
        if split=='train':
            y_true, y_pred, kwargs = self.predict(self.X, self.y, **kwargs)
        elif split=='test':
            y_true, y_pred, kwargs = self.predict(kwargs['X_TEST'], kwargs['Y_TEST'], **kwargs)
        else:
            ValueError('Unrecognized split: should be either train or test')
        # Get plots
        model = self.steps[-1]
        plots = model.performance_summary(y_true, y_pred)
        plt.show()
        return

    def save(self, save_path=None, **params):
        # Make sure save_path is a folder
        if save_path is None:
            save_path = LOCAL_MODEL_STORE
        else:
            save_path, _ = path.splitext(save_path)

        # Make sure it is empty
        assert path.isdir(save_path)
        makedirs(save_path, exist_ok=True)

        # Gen pipeline name
        dtm = datetime.datetime.now()
        suffix = serialize_dictionary(params['FILTER_CRITERIA'])
        pipe_name = 'pipeline_%s_%i%i%i-%i%i%i_%s' % \
                    (self.pipeline_type, dtm.day, dtm.month, dtm.year, dtm.hour, dtm.minute, dtm.second, suffix)
        full_path = path.join(save_path, pipe_name)
        makedirs(full_path)

        # Save pipeline variables
        pipe_vars = path.join(full_path, 'pipeline_variables.pkl')
        with open(pipe_vars, 'wb') as f:
            pickle.dump([self.pipeline_type,
                         self.is_fit,
                         #self.X,
                         #self.y,
                         #self.hyperparameter_range,
                         params
                         ], f)

        # Save blocks
        for x_block, i_block in enumerate(self.steps):
            # Get model type
            blk_type = i_block.__class__.__bases__[0].__name__
            blk_name = i_block.__class__.__name__
            # Save path
            block_path = path.join(full_path, 'block_%i_%s_%s' % (x_block, blk_type, blk_name))
            i_block.save(block_path, **params)

        # Log to MLflow
        environ["MLFLOW_TRACKING_USERNAME"] = "youbuntu"
        environ["MLFLOW_TRACKING_PASSWORD"] = "Dr!ss0un0urs"

        """
        import pandas as pd
        from sklearn import datasets
        from sklearn.ensemble import RandomForestClassifier
        import mlflow
        import mlflow.sklearn
        from mlflow.models.signature import infer_signature

        iris = datasets.load_iris()
        iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
        clf = RandomForestClassifier(max_depth=7, random_state=0)
        clf.fit(iris_train, iris.target)
        signature = infer_signature(iris_train, clf.predict(iris_train))
        mlflow.sklearn.log_model(clf, "dumbest model ever3", signature=signature)
        """
        log_model(
            artifact_path=pipe_name,
            python_model=PipelineWrapper(),
            code_path=[PATH_TO_CODE],
            artifacts={"pipeline_full_path": full_path},
        )

        return full_path

    def load(self, **params):
        # Make sure folder exists
        assert "pipeline_path" in params.keys()
        assert path.isdir(params["pipeline_path"])
        # Re-load pipeline variables
        full_path = params["pipeline_path"]
        pipe_vars = path.join(full_path, 'pipeline_variables.pkl')
        with open(pipe_vars, 'rb') as f:
            #self.pipeline_type, self.is_fit, self.X, self.y, self.hyperparameter_range, params = pickle.load(f)
            self.pipeline_type, self.is_fit, params = pickle.load(f)
        # List blocks
        ls_files = listdir(full_path)
        fl_filtr = [('block_' in i_) for i_ in ls_files]
        ls_files = [i_ for i_,j_ in zip(ls_files, fl_filtr) if j_]
        fl_order = [int(i_.split('_')[1]) for i_ in ls_files]
        for i_fl in range(len(ls_files)):
            i_file = ls_files[fl_order.index(i_fl)]
            i_path = path.join(full_path, i_file)
            # Determine block type
            blk_type = i_file.split('_')[2]
            if blk_type == 'TransformerBlock':
                i_block = TransformerBlock().load(i_path)
            elif blk_type == 'ModelBlock':
                i_block = ModelBlock().load(i_path)
            elif blk_type == 'PostprocessorBlock':
                i_block = PostprocessorBlock().load(i_path)
            else:
                raise ValueError('Unrecognized block type during pipeline loading')
            # Load block
            self.steps += [i_block]
        return self, params



# ======================================== #
# ------------ SPECIAL BLOCKS ------------ #
# ======================================== #
class DataLoaderBlock(TransformerBlock):

    def __init__(self):
        return

    def fit_transform(self, X, y=None, **fit_params):
        # X is assumed to be a relative path to a data file from the main data folder
        X = load_datafile(X, separator=DF_DATA_SEPARATOR, decimal=DF_DATA_DECIMAL)
        X.sort_values(by=['timestamp'], inplace=True)
        if y is not None:
            y = load_datafile(y, force_numeric=False, separator=DF_LABEL_SEPARATOR, decimal=DF_LABEL_DECIMAL)
            smpstart = get_sample_variables(y)[0]
            y.sort_values(by=[smpstart], inplace=True)
        return X.iloc[:20000], y

    def transform(self, X, y=None, **kwargs):
        return self.fit_transform(X, y, **kwargs)


class XYBlock(TransformerBlock):
    """
    This block turns dataframes X and y into numpy arrays
    It extracts the feature set and the targets to be predicted
    """
    def __init__(self, comment=''):
        self.comment = comment
        super(XYBlock).__init__()

    def fit_transform(self, X, y, **kwargs):
        # Make sure format is appropriate
        X_, y_ = check_input(X, y, x_type=list)
        # Make sure all elements have the same type
        elem_ar = [isinstance(i_, np.ndarray) for i_ in X_]
        elem_df = [isinstance(i_, pd.DataFrame) for i_ in X_]
        # Check input format
        if all(elem_df):
            # Convert all dataframes to arrays
            X_ = [i_.values for i_ in X_]
            y_ = [i_.values for i_ in y_]
            """
            # Check if elements can be merged along the first dimension
            i_shps = [np.shape(i_)[1:] for i_ in X]
            n_dims = [len(i_) for i_ in i_shps]
            if all_elements_equal(i_shps):
                X = np.concatenate(X, axis=0)
                y = np.concatenate(y, axis=0)
            else:
                raise ValueError('Unable to merge elements of the list')
            """
        elif all(elem_ar):
            # Make sure y is also array
            y_ = convert_input(y_, out_type=List[np.ndarray])
        else:
            raise ValueError('Input must be either list of dataframes or list of numpy arrays')
        return X_, y_, kwargs

    def transform(self, X, y, **kwargs):
        return self.fit_transform(X, y, **kwargs)


# ======================================== #
# --------------- WRAPPERS --------------- #
# ======================================== #
class PipelineWrapper(PythonModel):
    """
    Class to train and use FastText Models
    """

    def load_context(self, context):
        """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
        Args:
            context: MLflow context where the model artifact is stored.
        """
        params = {"pipeline_path": context.artifacts["pipeline_full_path"]}
        self.model, self.params = PipelineBlock([], load_mode=True).load(**params)

    def predict(self, context, model_input):
        """This is an abstract function. We customized it into a method to fetch the FastText model.
        Args:
            context ([type]): MLflow context where the model artifact is stored.
            model_input ([type]): the input data to fit into the model.
        Returns:
            [type]: the loaded model artifact.
        """
        return self.model, self.params
