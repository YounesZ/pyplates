# TODO CODE:
#   -   auto-add dataloader blocks and splitter blocks to pipelines
#   -   make example code for templates
#   -   make sure outputs X,y of a block have the same length
#   -   Add verbose mode for pipelines
#   -   Make an extract_time_series block and remove line from SeriesToTableBlock
#   -   Deal with the output of SeriesToTableBlock: not a dataframe
#   -   Add a variable name for clutch_speed_out and clutch_torque in the config file
#   -   correct bandpassfilter block for lists of dataframes

# TODO TESTS:
#   -   test pipelines for completeness
#   -   Make sure PRETRAINED_ENCODERS covers all FEATURES_TO_ENCODE

# TODO MODELLING:
#   -   Add cross-validation
#   -   Add hyperparameter tuning
#   -   Add tuning budget
#   -   Add fit results function
#   -   Split eval functions by RegressorBlock and ClassifierBlock
#   -   Add a check in pipeline for consistency between pipeline type and classifier type
#   -   Batches for BiLSTMClassifier are not randomized