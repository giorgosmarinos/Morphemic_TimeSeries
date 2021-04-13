import numpy as np
import pandas as pd
import os 

dataset_folder = "/home/jean-didier/Projects/morphemic/time-series-data/connected_consumer"
#dataset_folder = "/media/giwrikas/DATA/Morphemic_datasets"
#dataset_folder = "D:\\Morphemic_datasets\\"

def get_all_files():
    _files = []
    for root, dirs, files in os.walk(dataset_folder):
        for filename in files:
            _files.append(root +  '/' + filename)
    return _files 

def load_data():
    all_df = []
    for _file in get_all_files():
        all_df.append(pd.read_csv(_file,low_memory=False, error_bad_lines=False))
    return pd.concat(all_df, axis=0)

def percent_missing(data):
    percent_missing = data.isnull().sum() * 100 / len(data)
    missing_value_df = pd.DataFrame({'column_name': data.columns,
                                     'percent_missing': percent_missing})
    missing_value_df = missing_value_df.reset_index()
    missing_value_df = missing_value_df.drop(columns=['index'])
    missing_value_df.sort_values('percent_missing', inplace=True, ascending=False)
    print(missing_value_df) #TODO dont know for sure if it has to return something or just to print
    #return missing_value_df #if needed we can use return here



#TODO here has to be placed a function for handling missing data
def missing_data_handling(data ,drop_all_nan = False, fill_with_mean = False,
                          fill_with_median = False, rolling_mean = False, rolling_median = False):

    def drop_all_nan(data):
        data = data.dropna()
        return  data

    def fill_with_mean(data):
        # Filling using mean
        data = data.assign(FillMean=data.target.fillna(data.target.mean()))
        return data

    def fill_with_median(data):
        # Filling using median
        data = data.assign(FillMedian=data.target.fillna(data.target.median()))
        return data

    def rolling_mean(data):
        #Filling with rolling mean
        data = data.assign(RollingMean=0)  # To avoid pandas warning
        return data

    def rolling_median(data):
        #FIlling with rolling median
        data = data.assign(RollingMedian=0)
        return data

    def linear_inerpolation(data):
        data.assign(InterpolateLinear=data.target.interpolate(method='linear'))
        return data

    def time_interpolation(data): #For the time interpolation to succeed, the dataframe must have the index in Date format with intervals of 1 day or more, (daily, monthly, …) however, it will not work for time-based data, like hourly data and so.
        data.assign(InterpolateTime=data.target.interpolate(method='time'))
        return data

    def quadratic_interpolation(data):
        data.assign(InterpolateQuadratic=data.target.interpolate(method='quadratic'))
        return data

    def cubic_interpolation(data):
        data.assign(InterpolateCubic=data.target.interpolate(method='cubic'))
        return data

    def slinear_interpolation(data):
        data.assign(InterpolateSLinear=data.target.interpolate(method='slinear'))
        return data

    def akima_interpolation(data):
        data.assign(InterpolateAkima=data.target.interpolate(method='akima'))
        return data

    def polynomial_interpolation5(data):
        data.assign(InterpolatePoly5=data.target.interpolate(method='polynomial', order=5))
        return data

    def polynomial_interpolation7(data):
        data.assign(InterpolatePoly7=data.target.interpolate(method='polynomial', order=7))
        return data

    def spline_interpolate3(data):
        data.assign(InterpolateSpline3=data.target.interpolate(method='spline', order=3))
        return data

    def spline_interpolate4(data):
        data.assign(InterpolateSpline4=data.target.interpolate(method='spline', order=4))
        return data

    def spline_interpolate5(data):
        data.assign(InterpolateSpline4=data.target.interpolate(method='spline', order=5))
        return data


    if drop_all_nan == True:
        data = drop_all_nan(data)
    elif fill_with_mean == True:
        data = fill_with_mean(data)
    elif fill_with_median == True:
        data = fill_with_median(data)
    elif rolling_mean == True:
        data = rolling_mean(data)
    elif rolling_median == True:
        data = rolling_median(data)
    elif linear_inerpolation == True:
        data = linear_inerpolation(data)
    elif time_interpolation == True:
        data = time_interpolation(data)
    elif quadratic_interpolation == True:
        data = quadratic_interpolation(data)
    elif cubic_interpolation == True:
        data = cubic_interpolation(data)
    elif slinear_interpolation == True:
        data = slinear_interpolation(data)
    elif akima_interpolation == True:
        data = akima_interpolation(data)
    elif polynomial_interpolation5 ==True:
        data = polynomial_interpolation5(data)
    elif polynomial_interpolation7 == True:
        data = polynomial_interpolation7(data)
    elif spline_interpolate3 ==True:
        data = spline_interpolate3(data)
    elif spline_interpolate4 == True:
        data = spline_interpolate4(data)
    elif spline_interpolate5 == True:
        data = spline_interpolate5(data)
    else:
        pass

    return data



def datetime_conversion(data, column_name):
    data[column_name] = pd.to_datetime(data[column_name], unit='s')
    data = data.set_index(column_name)
    return data


list_of_important_features = []
def important_data(data, list_of_important_features):
    data_ = data[list_of_important_features]
    return data_


def resample(data, rate='360S'):
    resampled_data= data.resample(rate).mean() #TODO maybe the dot in data_. will cause problems
    return resampled_data


def resample_median(data):
    exogenous_data = data.resample('360S').median()
    return exogenous_data


def data_without_resampling(data_, feature_of_interest):
    y = data_[feature_of_interest]
    X = data_.drop(columns=feature_of_interest)
    return  X, y


