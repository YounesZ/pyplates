import mlflow
from pyplates.config import MLFLOW_EXCLUDE_PARAMS

def get_mlflow_experiment_id(experiment_name):
    elist = mlflow.list_experiments()
    eid = [i_.experiment_id for i_ in elist if i_.name == experiment_name]
    if len(eid) > 0:
        eid = eid[0]
        mlflow.set_experiment(experiment_name)
    else:
        eid = mlflow.create_experiment(experiment_name)
    return eid

def mlflow_log_params(params):
    # List parameters
    names = list( params.keys() )
    for i_ in MLFLOW_EXCLUDE_PARAMS:
        if i_ in names:
            names.pop(names.index(i_))
    # Log them
    for i_ in names:
        mlflow.log_param(i_, params[i_])

