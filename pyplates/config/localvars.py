from os import path

# Make code path
PATH_TO_CODE = __file__
idx = PATH_TO_CODE.index(path.join("config", "localvars.py"))
PATH_TO_CODE = PATH_TO_CODE[:idx-1]