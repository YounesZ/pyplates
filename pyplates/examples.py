from ml_pipelines.Templates.config import read_default_config
from ml_pipelines.Templates.main_blocks import PipelineBlock
from ml_pipelines.Templates.model_blocks import RegressorBlock, ClassifierBlock

params = read_default_config()


def pipeline_kisspoint_prediction_with_encoder():
    # Custom hyperparameters
    params['PIPELINE_TYPE'] = 'regression'
    params['DATA_FILE'] = 'ClutchEngagment_FullDataset.csv'
    params['LABEL_FILE'] = 'ClutchingsInfo.csv'
    params['TARGET_COLUMN'] ='kissPoint'

    # Make list of blocks
    ls_blocks = [('Signal epoching', SignalEpochsBlock()),
                 ('Equalize epochs length', EpochsEqualizerBlock()),
                 ('Make slip signal', SlipSignalBlock()),
                 ('Feature union block', FeatureUnionBlock([('Feature encoding', FeaturesEncoderBlock()),
                                                            ('Feature averaging', FeaturesToScalarBlock())])),
                 ('Extract target column', ExtractFeaturesTargetBlock()),
                 ('Decision tree regressor', RegressorBlock(regressor_name="DecisionTreeRegressor"))]

    # Make & use pipeline
    pipe = PipelineBlock(ls_blocks, **params)
    train_results = pipe.train(**params)
    pipe.performance_summary(**params)


def pipeline_kisspoint_prediction_rnn():
    # Custom hyperparameters
    params['PIPELINE_TYPE'] = 'classification'
    params['DATA_FILE'] = 'ClutchEngagment_FullDataset.csv'
    params['LABEL_FILE'] = 'ClutchingsInfo.csv'
    params['TARGET_COLUMN'] ='kissPoint'
    params['FEATURES_SCALAR']  = []
    params['FEATURES_TIME_SERIES'] = ["clutch_hirev_speed_out", "clutch_hirev_slip_speed",
                                      "clutch_low_slip", "clutch_low_slip_speed", "clutch_low_speed_out",
                                      "clutch_rev_slip", "clutch_speed_in", "cvt_ratio_actual", "cvt_speed_in",
                                      "cvt_speed_out", "intshaft_speed", "temp_sump", "temp_inlet"]

    # Make list of blocks
    ls_blocks = [('Clean data', DataCleanerBlock(**params)),
                 ('Signal epoching', SignalEpochsBlock(**params)),
                 ('Min max scaling', MinMaxScalerBlock(**params)),
                 ('Extract target column', ExtractFeaturesTargetBlock(**params)),
                 ('Transform y-target to y-series', TargetToSeriesBlock(**params)),
                 ('Series to table block', SeriesToTableBlock(**params)),
                 ('Bidirectional LSTM', ClassifierBlock(classifier_name="BiLSTMClassifier"))]

    # Make & use pipeline
    pipe = PipelineBlock(ls_blocks, **params)
    train_results = pipe.train(**params)
    pipe.performance_summary(**params)


def pipeline_kisspoint_prediction_rnn2():

    # Custom hyperparameters
    params['PIPELINE_TYPE'] = 'classification'
    params['DATA_FILE'] = 'ClutchEngagment_FullDataset.csv' # 'ClutchEngagment_FullDataset.csv'
    params['LABEL_FILE'] = 'ClutchingsInfo.csv'
    params['TARGET_COLUMN'] ='kissPoint'
    params['FEATURES_TIME_SERIES'] = ["clutch_hirev_speed_out", "clutch_hirev_slip_speed",
                                      "clutch_low_slip", "clutch_low_slip_speed", "clutch_low_speed_out",
                                      "clutch_rev_slip", "clutch_speed_in", "cvt_ratio_actual", "cvt_speed_in",
                                      "cvt_speed_out", "intshaft_speed", "temp_sump", "temp_inlet"]

    # Make list of blocks
    ls_blocks = [('Clean data', DataCleanerBlock(**params)),
                 ('Signal epoching', SignalEpochsBlock(**params)),
                 ('Min max scaling', MinMaxScalerBlock(**params)),
                 ('Extract target column', ExtractTargetBlock(**params)),
                 ('Transform y-target to y-series', TargetToSeriesBlock(**params)),
                 ('Series to table block', SeriesToTableBlock(**params)),
                 ('Bidirectional LSTM', ClassifierBlock(classifier_name="BiLSTMClassifier"))]

    # Make & use pipeline
    pipe = PipelineBlock(ls_blocks, **params)
    train_results = pipe.train(**params)
    pipe.performance_summary(**params)


def pipeline_virtual_torquemeter():
    # Custom hyperparameters
    params['PIPELINE_TYPE'] = 'regression'
    params['DATA_FILE'] = 'ClutchEngagment_FullDataset.csv'
    params['LABEL_FILE'] = 'SlippingInfo.csv'
    params['TARGET_COLUMN'] = 'clutch_torque'
    params['FEATURES_SCALAR'] = []
    params['FEATURES_TIME_SERIES'] = ['cvt_speed_in', 'cvt_speed_out', 'temp_sump', 'temp_inlet', 'clutch_speed_in',
                                     'clutch_speed_out', 'fuel_pedal_pos', 'engine_torque', 'intshaft_speed', 'trans_speed_out']

    # Make list of blocks
    ls_blocks = [('Clean data', DataCleanerBlock(**params)),
                 ('Band-pass filtering', BPfilterBlock(**params)),
                 ('Signal epoching', SignalEpochsBlock(**params)),
                 ('Make clutch torque signal', ClutchTorqueSignalBlock(**params)),
                 ('Extract target column', ExtractFeaturesTargetBlock(**params)),
                 ('Series to table block', SeriesToTableBlock(**params)),
                 ('Bidirectional LSTM regressor', RegressorBlock(regressor_name="BiLSTMRegressor"))]

    # Make & use pipeline
    pipe = PipelineBlock(ls_blocks, **params)
    train_results = pipe.train(**params)
    pipe.performance_summary(**params)




if __name__ == '__main__':
    #pipeline_kisspoint_prediction_with_encoder() # Younes' pipeline
    pipeline_kisspoint_prediction_rnn2() # Alex's pipeline
    #pipeline_virtual_torquemeter()  # Xingshuai's pipeline