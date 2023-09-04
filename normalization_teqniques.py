import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn import preprocessing
import math


def get_normalize_functions(training_data):
    normalizers=[]
    training_data_Numbers = pd.get_dummies(
        training_data.select_dtypes(include=[np.number]))  # extract non-string
    training_data_Strings = training_data[
        training_data.columns.difference(list(training_data_Numbers))]  # extract string


    # Z-SCORE
    training_data_zScore = zscore(training_data_Numbers)
    training_data_zScore = pd.concat([training_data_zScore, training_data_Strings], axis="columns")
    normalizers.append('Z-Score')

    # MIN-MAX
    min_max_scaler = preprocessing.MinMaxScaler()
    training_data_MINMAX = min_max_scaler.fit_transform(training_data_Numbers)
    training_data_MINMAX = pd.DataFrame(training_data_MINMAX, columns =list(training_data_Numbers))
    training_data_MINMAX = pd.concat([training_data_MINMAX, training_data_Strings], axis="columns")
    normalizers.append('Min-Max')

    # MAXABS
    max_abs_scaler = preprocessing.MaxAbsScaler()
    training_data_MaxAbs = max_abs_scaler.fit_transform(training_data_Numbers)
    training_data_MaxAbs = pd.DataFrame(training_data_MaxAbs, columns=list(training_data_Numbers))
    training_data_MaxAbs = pd.concat([training_data_MaxAbs, training_data_Strings], axis="columns")
    normalizers.append('Max-Abs')

    ## Robust
    rbs = preprocessing.RobustScaler()
    training_data_robust = rbs.fit_transform(training_data_Numbers)
    training_data_robust = pd.DataFrame(training_data_robust, columns=list(training_data_Numbers))
    training_data_robust = pd.concat([training_data_robust, training_data_Strings], axis="columns")
    normalizers.append('Robust')

    return normalizers, [training_data_zScore, training_data_MINMAX,training_data_MaxAbs,training_data_robust]
