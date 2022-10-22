import numpy as np
import pandas as pd
from src.utils import check_type
#from pyplates.utils import define_hyperparameter_range
from sklearn.preprocessing import MinMaxScaler
from pyplates.main_blocks import TransformerBlock
from pytox.utils.preprocessing import bpfilter_columns
from pytox.utils.data_structures import check_input
from pytox.utils.signal_processing import pad_series,\
                                          closest_index_to, \
                                          make_shifted_series, \
                                          get_sample_variables, \
                                          ts_custom_split
from pytox.unsupervised.data_encoding import encode_dataset_univariate



# ================================= #
# -------- DATA STRUCTURES -------- #
# ================================= #
class SeriesToTableBlock(TransformerBlock):

    def __init__(self, **kwargs):
        self.is_fit = False
        if 'N_SHIFTS_RECURRENT_MODELLING' in kwargs:
            self.n_history = kwargs['N_SHIFTS_RECURRENT_MODELLING']
        else:
            raise ValueError('Argument N_SHIFTS_RECURRENT_MODELLING not passed during block initialization')

    def fit_transform(self, X, y=None, **fit_params):
        X_, y_ = [], []
        for i_,j_ in zip(X, y):
            xx, yy = make_shifted_series(i_[fit_params['FEATURES_TIME_SERIES']], j_, self.n_history)
            X_ += [xx]
            y_ += [yy]
        return X_, y_

    def transform(self, X, y=None, **kwargs):
        return self.fit_transform(X, y, **kwargs)

class SignalEpochsBlock(TransformerBlock):

    def __init__(self, **kwargs):
        self.is_fit = False
        return

    def fit_transform(self, X, y=None, min_samples=10, **kwargs):
        X_, _ = check_input(X, x_type=pd.DataFrame)
        varstart, varstop = get_sample_variables(y)
        # Make sure index is timestamp
        X_ = X_.reset_index().set_index('timestamp')
        # make data matrix
        mtx = []
        for i_sh in range(len(y)):
            # Get shift limits
            idsh = y.iloc[i_sh]
            # Slice the dataframe
            slc = X_.loc[idsh[varstart]:idsh[varstop]]
            mtx += [slc]

        # --- Filter out near-empty epochs
        # Remove small epochs
        id_keep = [x_ for x_,i_ in enumerate(mtx) if np.shape(i_)[0]>min_samples]
        mtx_nem = [mtx[i_] for i_ in id_keep]
        y_nem = y.iloc[id_keep]
        self.is_fit = True
        return mtx_nem, y_nem

    def transform(self, X, y=None, **kwargs):
        return self.fit_transform(X, y, **kwargs)

class EpochsEqualizerBlock(TransformerBlock):

    def __init__(self, **kwargs):
        return

    def fit_transform(self, X, y=None, **fit_params):

        # Check params
        nb_samples = 'min'
        if 'nb_samples' in fit_params:
            ixval = fit_params.index('nb_samples')
            min_samples = ixval+1
        if nb_samples=='min':
            nb_samples = np.min([i_.shape[0] for i_ in X])
        else:
            raise NotImplementedError('Cannot equalize epochs using the value %s for parameter %s' %
                                      (nb_samples, 'nb_samples'))

        # Equalize epochs
        equalized = []
        for i_ep in X:
            n_miss = i_ep.shape[0] - nb_samples
            if n_miss < 0:      # epoch is too short
                equalized += [pad_series(i_ep, nb_samples)]
            elif n_miss > 0:    # epoch is too long
                equalized += [i_ep.iloc[-nb_samples:]]
            else:
                equalized += [i_ep]
        return equalized, y

    def transform(self, X, y=None, **kwargs):
        return self.fit_transform(X, y, **kwargs)

class FeatureUnionBlock(TransformerBlock):

    def __init__(self, blocks_list, **kwargs):
        self.blocks_list = blocks_list
        return

    def fit_transform(self, X, y=None, **fit_params):
        Xc, yc = [], []
        # Loop on blocks
        for i_ in self.blocks_list:
            X_, y_ = i_[1].fit_transform(X, y, **fit_params)
            Xc += [X_]
            yc += [y_]

        # Concat dataframes
        Xc = np.hstack(Xc)
        return Xc, y

    def transform(self, X, y=None, **kwargs):
        Xc, yc = [], []
        # Loop on blocks
        for i_ in self.blocks_list:
            X_, y_ = i_[1].transform(X, y, **kwargs)
            Xc += [X_]
            yc += [y_]

        # Concat dataframes
        Xc = np.hstack(Xc)
        return Xc, y


# ================================= #
# --------- PREPROCESSING --------- #
# ================================= #
class DataCleanerBlock(TransformerBlock):

    def __init__(self, **kwargs):
        self.is_fit = False

    def fit_transform(self, X, y=None, **fit_params):
        # Fill nans
        X, _ = check_input(X, x_type=pd.DataFrame)
        X_ = X.fillna(axis='columns', method='ffill')
        self.is_fit = True
        return X_, y

    def transform(self, X, y=None, **kwargs):
        return self.fit_transform(X, y, **kwargs)

class MinMaxScalerBlock(TransformerBlock):

    def __init__(self, **kwargs):
        self.is_fit = False
        self.fittable_columns = kwargs['FEATURES_TIME_SERIES']
        self.model = MinMaxScaler()
        super().__init__()

    def fit_transform(self, X, y=None, **fit_params):
        # Apply
        X_, _ = check_input(X)
        for i_ in X_:
            i_[self.fittable_columns] = self.model.fit(i_[self.fittable_columns]).transform(i_[self.fittable_columns])
        return X_, y

    def transform(self, X, y=None, **kwargs):
        # Apply
        X_, _ = check_input(X)
        for i_ in X_:
            i_[self.fittable_columns] = self.model.transform(i_[self.fittable_columns])
        return X_, y

class BPfilterBlock(TransformerBlock):

    def __init__(self, filter_order=8, window_size=0.2, filter_type='lowpass', **kwargs):
        # Set hyperparameter range
        hyper_range = {'filter_order': {"default": filter_order, "range": [2, 12]},
                       'window_size': {"default": window_size, "range": [0.1, 0.3]},
                       'filter_type': {"default": filter_type}}
        super().__init__(hyper_range)
        return

    def fit_transform(self, X, y=None, **fit_params):
        # Process training data
        check_input(X, x_type=pd.DataFrame)
        X = bpfilter_columns(X, self.filter_order, self.window_size, self.filter_type)
        return X, y

    def transform(self, X, y, **kwargs):
        return self.fit_transform(X, y, **kwargs)


# ================================= #
# ------ FEATURE ENGINEERING ------ #
# ================================= #
class FeatureEngineeringBlock(TransformerBlock):

    def __init__(self):
        self.fit_state = False
        self.mapping_dict = {}
        self.encoding_dict = {}

    def fit_transform(self, X, y=None, mapping_dict=None, encoding_dict={}, transform_only=False, **fit_params):
        """
        X:  data processed
        """
        check_type(X, DataFrame)
        # --- Feature engineering
        # Turn car year to age
        dtc = convert_car_age(X)
        # Fill missing body type field
        dtc = fill_missing_bodytype(dtc)
        # Map categories: color, condition, fuel, body, trans, wheel
        dtc, mapping_dict = map_categories(dtc, mapping_dict)
        # One-hot encode categorical features
        dtc, encoding_dict = onehotencode_categorical(dtc, encoding_dict)

        self.fit_state = True
        self.data_engineered = dtc
        if not transform_only:
            self.mapping_dict = mapping_dict
            self.encoding_dict = encoding_dict
        return dtc, y

    def transform(self, X, y=None, **kwargs):
        """
        X:  data loaded
        """
        if not self.fit_state:
            raise ValueError('Cannot run transform before fitting the transformer')
        data_engineered, y = self.fit_transform(X, y, transform_only=True, **kwargs)
        return data_engineered, y

class TargetToSeriesBlock(TransformerBlock):

    def __init__(self, **kwargs):
        self.is_fit = False

    def fit_transform(self, X, y=None, **fit_params):
        # Check input
        X, y = check_input(X, y)
        # Turn y into series
        y_ = []
        for i_,j_ in zip( range(len(X)), y ):
            srs = pd.Series([0] * len(X[i_]), index=X[i_].index)
            ilc = closest_index_to(srs, srs.index[0]+j_)
            srs.loc[ilc] = 1
            y_ += [srs]
        return X, y_

    def transform(self, X, y=None, **kwargs):
        return self.fit_transform(X, y, **kwargs)

class SlipSignalBlock(TransformerBlock):

    def __init__(self, **kwargs):
        self.feature_name = 'clutch_slip'
        return

    def fit_transform(self, X, y=None, **fit_params):
        # Loop on epochs - Check which clutch is engaged
        for i_ in range(len(y)):
            if "2Rev" in y.iloc[i_].EngagementType:  # Rev clutch was engaged
                X[i_][self.feature_name] = X[i_][HIREV_CLUTCH_NAME]
            else:
                X[i_][self.feature_name] = X[i_][LOW_CLUTCH_NAME]
        return X, y

    def transform(self, X, y=None, **kwargs):
        return self.fit_transform(X, y, **kwargs)

class ClutchTorqueSignalBlock(TransformerBlock):

    def __init__(self, **kwargs):
        self.feature_name = 'clutch_torque'

    def fit_transform(self, X, y=None, **fit_params):
        check_input(X)
        for i_,(_,j_) in zip(X, y.iterrows()):
            if j_.clutch == 'low':
                i_['clutch_speed_out'] = i_.clutch_low_speed_out
                i_[self.feature_name] = i_.clutch_low_torque
            elif j_.clutch == 'hirev':
                i_['clutch_speed_out'] = i_.clutch_hirev_speed_out
                i_[self.feature_name] = i_.clutch_hirev_torque
            else:
                raise ValueError('Unrecognized slip type %s' % j_.clutch)
        return X, y

    def transform(self, X, y=None, **kwargs):
        return self.fit_transform(X, y, **kwargs)

class FeaturesToScalarBlock(TransformerBlock):

    def __init__(self, **kwargs):
        return

    def fit_transform(self, X, y=None, **fit_params):
        # Get dataframe slice
        X_slice = [i_[fit_params['FEATURES_SCALAR']].mean() for i_ in X]
        X_slice = np.array(X_slice)
        return X_slice, y

    def transform(self, X, y=None, **kwargs):
        return self.fit_transform(X, y, **kwargs)
    
class FilterCriteriaBlock(TransformerBlock):
    
    def __init__(self, criteria, **kwarga):
        self.criteria =  criteria
        super().__init__()
        
    def fit_transform(self, X, y=None, new_criteria=None, **kwargs):
        # Feature selection
        if new_criteria is None:
            criteria = self.criteria
        # Apply filtering
        filtered = filter_by_features(data_engineered, criteria)
        return filtered, y

    def transform(self, X, y, new_criteria=None, **kwargs):
        return self.fit_transform(X, y, new_criteria, **kwargs)
        

# ================================= #
# ----------- MODELLING ----------- #
# ================================= #
class ExtractFeaturesTargetBlock(TransformerBlock):

    def __init__(self, **kwargs):
        return

    def fit_transform(self, X, y=None, **fit_params):
        # --- Process X, y
        X_, y_ = check_input(X, y, x_type=list)
        i_target = fit_params['TARGET_COLUMN']
        # Get target
        if (y_ is not None) and all(isinstance(i_, pd.DataFrame) for i_ in y_) and all(i_target in i_.columns for i_ in y_):    # Target was found in y_
            y_ = [i_[i_target] for i_ in y_]
        elif all(i_target in i_.columns for i_ in X_):  # Target was found in X_
            y_ = [i_[i_target] for i_ in X_]
        else:
            raise ValueError('Cannot find target in X or y')
        # Get rid of target in the features matrix 
        accessible_features = fit_params['FEATURES_COLUMNS']
        if fit_params['TARGET_COLUMN'] in accessible_features:
            ixpop = accessible_features.index(fit_params['TARGET_COLUMN'])
            accessible_features.pop(ixpop)
        # Get features
        if isinstance(X_[0], pd.DataFrame):
            X_ = [i_[accessible_features] for i_ in X_]
        return X_, y_, fit_params

    def transform(self, X, y=None, **kwargs):
        return self.fit_transform(X, y, **kwargs)

class FeaturesEncoderBlock(TransformerBlock):

    def __init__(self, **kwargs):
        return

    def fit_transform(self, X, y=None, **fit_params):
        # Encode data
        features_enc = encode_dataset_univariate(X, PRETRAINED_ENCODERS, features_to_encode=fit_params['FEATURES_TIME_SERIES'])
        return features_enc, y

    def transform(self, X, y=None, **kwargs):
        return self.fit_transform(X, y, **kwargs)

class ExtractTargetBlock(TransformerBlock):

    def __init__(self, **kwargs):
        return

    def fit_transform(self, X, y=None, **fit_params):
        # Get target
        y = y[fit_params['TARGET_COLUMN']]
        return X, y

    def transform(self, X, y=None, **kwargs):
        return self.fit_transform(X, y, **kwargs)

class SplitFeaturesTargetBlock(TransformerBlock):

    def __init__(self, **kwargs):
        return

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit_transform(X, y, **fit_params)

    def transform(self, X, y=None, **params):
        # --- Check for parameter value
        if 'ENCODING_INFO' in params.keys():
           info = params['ENCODING_INFO']
        else:
            raise ValueError('Missing encoding info from the parameters dictionary passed to SplitFeaturesTargetBlock')

        # --- Map categorical features to onehot
        col_names = []
        for i_ in FEATURES:
            re_named = i_
            if i_ in info['categorical_features']:
                # Remap name
                ft_index = info['categorical_features'].index(i_)
                re_named = info['extended_names'][ft_index]
            # Add to the new columns names
            col_names += [re_named]
        # Format names as 1D vector
        col_names = list( np.hstack(col_names) )
        X = dtc[col_names]

        # --- Map categorical targets to onehot
        re_named = TARGET
        if TARGET in info['categorical_features']:
            # Remap name
            ft_index = info['categorical_features'].index(TARGET)
            re_named = info['extended_names'][ft_index]
        y = dtc[re_named]
        return X, y

class TestTrainSplitterBlock(TransformerBlock):

    def __init__(self, comment='', **kwargs):
        self.comment = comment

    def fit_transform(self, X, y=None, **fit_params):
        # Split X, y separately
        X_train, X_test, svec = ts_custom_split(X, test=fit_params['TEST_FRACTION'], how='shuffled', random_seed=fit_params['RANDOM_SEED'])
        y_train, y_test, _ = ts_custom_split(y, svec=svec, test=fit_params['TEST_FRACTION'], how='shuffled', random_seed=fit_params['RANDOM_SEED'])
        fit_params['X_TEST'] = X_test
        fit_params['Y_TEST'] = y_test
        return X_train, y_train, fit_params

    def transform(self, X, y=None, **kwargs):
        print('WARNING: The %s block is only valid for training pipeline, ignoring ...' % self.__class__.__name__)
        return X, y, kwargs
