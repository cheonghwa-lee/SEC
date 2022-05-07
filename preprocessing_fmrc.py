import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import copy

# 1. exploratory data analysis
def EDA(obj):
    """
    1. basic statistics
    """
    print(obj.shape)
    print(obj.info())
    for col in obj.columns:
        print("-" * 100)
        print(col)
        print('{}\n'.format(obj[col].head()))
        print('{}\n'.format(obj[col].describe()))
        # print('{}\n'.format(obj[col].unique()))
        print("unique: ", len(pd.unique(obj[col])), pd.unique(obj[col]))
        print("-" * 100)
    visualization(obj)

def visualization(obj):
    """
    1. plot
    2. histogram
    3. line
    4. scatter
    """
    x = list(range(0,obj.shape[0]))
    for col in obj.columns:
        print('-' * 50)
        print('col: ', col)
        plt.plot(x, obj[col])
        # plt.show()

# 2. pre_processing
def pre_processing_obu(obj):
    """
    1. column
    2. row
    # 3. sort
    """
    # column
    obj = uniqueness(obj)
    # obj = no_meaning_column(obj)

    # row
    obj = rsu_latlon_nan_elimination(obj)
    obj = latlon_range_outlier_elimination(obj, "latitude", "longitude")
    
    # sort
    # obj = sorting_by_time(obj)

    return obj

def pre_processing_rsu1(obj):
    # column
    obj = uniqueness(obj)
    # obj = no_meaning_column(obj)
    # row
    obj = rsu_latlon_nan_elimination(obj)
    obj = latlon_range_outlier_elimination(obj, "detect_latitude", "detect_longitude")
    # sort
    # obj = sorting_by_time(obj)

    return obj

def pre_processing_rsu2(obj):
    # column
    obj = uniqueness(obj)
    # obj = no_meaning_column(obj)
    # row
    obj = rsu_latlon_nan_elimination(obj)
    obj = latlon_range_outlier_elimination(obj, "detect_latitude", "detect_longitude")
    # sort
    # obj = sorting_by_time(obj)

    return obj

def uniqueness(obj):
    """
    """
    obj_trivial = copy.deepcopy(obj)
    for col in obj.columns:
        uniq = np.unique(obj[col].astype(str))
        print('=' * 50)
        print("# col {}, n_uniq {}, uniq {}".format(col, len(uniq), uniq))
        if len(uniq) == 1:
            del obj[col]
        else:
            del obj_trivial[col]
    print(obj.shape, obj_trivial.shape)
    print(obj.columns)
    print(obj_trivial.columns)

    return obj

def rsu_latlon_nan_elimination(obj):
    obj = obj.dropna(axis=0)
    return obj

def latlon_range_outlier_elimination(obj, col1, col2):
    DEN = 10000000
    del_idx = []
    for row in obj.index:
        lat = obj[col1][row]/DEN
        lon = obj[col2][row]/DEN
        if lat > 40 or lat < 30 or lon > 130 or lon < 0:
            del_idx.append(row)
    obj = obj.drop(index=del_idx)
    return obj

"""
def omission(obj):
    print(obj.isnull().sum())

def sorting_by_time(obj, col):
    obj = obj.sort_values(by=["msg_received_time"], acending=[True])
"""
