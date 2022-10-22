import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, ElasticNet, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from pyplates.main_blocks import ModelBlock
from pytox.networks.recurrent import BiLSTMClassifier
from pytox.networks.recurrent import BiLSTMRegressor
from pytox.utils.visualization import plot_averaged_RMSE,\
                                      plot_single_prediction,\
                                      scatter_real_predicted, \
                                      histogram_prediction_errors


# --- MODELLING
LIST_CLASSIFIERS = {"BiLSTMClassifier": {'class': BiLSTMClassifier,
                                         'type': "classifier",
                                         'init_args': {},
                                         'display': "bidirectional LSTM",
                                         'hyper': {'layers': list(range(1,6)), 'k_neurons': list(range(3,13)), 'batch_size': [4, 8, 16, 32, 128], 'n_iterations': [1, 2, 4, 8, 16]},
                                         }
                    }
LIST_REGRESSORS = {"LGBMRegressor": {'class': LGBMRegressor,
                                     'type': 'regressor',
                                     'init_args': {},
                                     'display': 'Light GBM',
                                     'hyper':  {'num_leaves': [7, 14, 21, 28, 31, 50], 'learning_rate': [0.1, 0.03, 0.003], 'max_depth': [-1, 3, 5], 'n_estimators': [50, 100, 200, 500]},
                                     },
                   "CatBoostRegressor": {'class': CatBoostRegressor,
                                         'type': 'regressor',
                                         'init_args': {},
                                         'display': 'CatBoost regressor',
                                         'hyper': {'learning_rate': [0.03, 0.1],
                                                   'depth': [4, 6, 10],
                                                   'l2_leaf_reg': [1, 3, 5, 7, 9]}
                                         },
                   "SVR": {'class': SVR,
                           'type': 'regressor',
                           'init_args': {},
                           'display': 'Support vector regressor',
                           'hyper': {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                                     'C': [1,5,10],
                                     'degree': [3,8],
                                     'coef0': [0.01,10,0.5],
                                     'gamma': ('auto','scale')}
                           },
                   "XGBRegressor": {'class': XGBRegressor,
                                    'type': 'regressor',
                                    'init_args': {},
                                    'display': 'XGBoost regressor',
                                    'hyper': {'min_child_weight':[4,5],
                                              'gamma':[i/10.0 for i in range(3,6)],
                                              'subsample':[i/10.0 for i in range(6,11)],
                                              'colsample_bytree':[i/10.0 for i in range(6,11)],
                                              'max_depth': [2,3,4]}
                                    },
                   "GradientBoostingRegressor": {'class': GradientBoostingRegressor,
                                                 'type': 'regressor',
                                                 'init_args': {},
                                                 'display': 'Gradient boosting regressor',
                                                 'hyper': {'n_estimators':[100,250, 500],
                                                           'learning_rate': [0.1,0.05,0.02],
                                                           'min_samples_leaf':[3],
                                                           'max_features':[1.0]}
                                                 },
                   "LinearRegression": {'class': LinearRegression,
                                        'type': 'regressor',
                                        'init_args': {},
                                        'display': 'Linear regressor',
                                        'hyper': {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                                  "fit_intercept": [True, False],
                                                  "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                                                  }
                                        },
                   "SGDRegressor": {'class': SGDRegressor,
                                    'type': 'regressor',
                                    'init_args': {},
                                    'display': 'Stochastic gradient descent',
                                    'hyper': {'alpha': 10.0 ** -np.arange(1, 7),
                                              'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
                                              'penalty': ['l2', 'l1', 'elasticnet'],
                                              'learning_rate': ['constant', 'optimal', 'invscaling']}
                                    },
                   "KernelRidge": {'class': KernelRidge,
                                   'type': 'regressor',
                                   'init_args': {'kernel': "rbf", 'gamma':0.1},
                                   'display': 'Linear support vector',
                                   'hyper': {'alpha': [1,0.1,0.01,0.001,0.0001,0] ,
                                             "fit_intercept": [True, False],
                                             "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
                                   },
                   "ElasticNet": {'class': ElasticNet,
                                  'type': 'regressor',
                                  'init_args': {},
                                  'display': 'Elasic Net',
                                  'hyper': {"alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0],
                                            'l1_ratio': np.arange(0, 1, 0.01)}
                                  },
                   "BayesianRidge": {'class': BayesianRidge,
                                     'type': 'regressor',
                                     'init_args': {},
                                     'display': 'Bagging',
                                     'hyper': {'alpha_init': [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.9],
                                               'lambda_init': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-9]}
                                     },
                   "DecisionTreeRegressor": {'class': DecisionTreeRegressor,
                                             'type': 'regressor',
                                             'init_args': {},
                                             'display': 'Bagging',
                                             'hyper': {"splitter":["best","random"],
                                                       "max_depth" : [1,3,5,7,9,11,12],
                                                       "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
                                                       "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                                       "max_features":["auto","log2","sqrt",None],
                                                       "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90]}
                                             },
                    "BiLSTMRegressor": {'class': BiLSTMRegressor,
                                        'type': 'regressor',
                                        'init_args': {},
                                        'display': 'Bidirectional LSTM regressor',
                                        'hyper': {"hidden_size":["best","random"],
                                                  "num_layer" : [1,3,5,7,9,11,12],
                                                  "dropout":[1,2,3,4,5,6,7,8,9,10],
                                                  "batch_first":[True, False],
                                                  "lr": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
                                                  "batch_size": [4, 8, 16, 32, 128]
                                                  }
                                        }
                   }


class RegressorBlock(ModelBlock):

    def __init__(self, regressor_name, comment='', **kwargs):
        self.comment = comment
        super().__init__()
        if regressor_name in LIST_REGRESSORS:
            self.model = LIST_REGRESSORS[regressor_name]["class"]
            self.hyperparameter_range = LIST_REGRESSORS[regressor_name]["hyper"]
        else:
            raise ValueError('Regressor %s not implemented' % regressor_name)
        return

    def performance_summary(self, y_true, y_pred, **kwargs):
        plots = []
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        scores= pd.DataFrame(data=[], columns=['RMSE', 'MAE'])
        # 1) Plot example of real vs predicted
        plot_single_prediction(y_true, y_pred, ax[0,0], **kwargs)
        # 2) Plot averaged RMSE across trials
        plot_averaged_RMSE(y_true, y_pred, ax[1,0], **kwargs)
        # 3) Plot avg real vs avg predicted (for peak)
        scatter_real_predicted(y_true, y_pred, ax[0,1], **kwargs)
        # 4) Plot error histogram
        histogram_prediction_errors(y_true, y_pred, ax[1,1], **kwargs)
        return plots, scores


class ClassifierBlock(ModelBlock):

    def __init__(self, classifier_name):
        if classifier_name in LIST_CLASSIFIERS:
            self.model = LIST_CLASSIFIERS[classifier_name]["class"]
            self.hyperparameter_range = LIST_CLASSIFIERS[classifier_name]["hyper"]
        else:
            raise ValueError('Classifier %s not implemented' % classifier_name)
        return

    def performance_summary(self, y_true, y_pred, **kwargs):
        plots = []
        scores= pd.DataFrame(data=[], columns=['accuracy', 'precision', 'recall', 'sensitivity', 'specificity'])
        # 1) Plot confusion matrix
        # 2) Compute scores
        return plots, scores
