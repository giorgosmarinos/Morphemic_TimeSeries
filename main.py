from pre_processing.preprocessing import load_data, percent_missing, datetime_conversion
from pre_processing.preprocessing import important_data, resample, resample_median, missing_data_handling
from pre_processing.Data_transformation import reshape_data_single_lag, series_to_supervised
from models.ML_models import LSTM_model
from plots.plots import plot_train_test_loss
from pre_processing.Data_transformation import predictions_and_scores, Min_max_scal
import matplotlib.pyplot as plt
import pandas as pd

metrics = ['performance','request_rate', 'cpu_usage', 'memory','served_request']

data = load_data()

data = data.round(decimals=2)

data = missing_data_handling(data, rolling_mean=True)

percent_missing(data)

data = datetime_conversion(data, 'time')

data = important_data(data, metrics)

data = resample(data)

data = Min_max_scal(data)
print(data)
data = series_to_supervised(data, 24, 1)
print(data)
train_X, train_y, test_X, test_y, val_X, val_y = reshape_data_single_lag(data,  0.6, 0.2, 0.2 )

model = LSTM_model(train_X, train_y, test_X, test_y)

model.summary()

plot_train_test_loss(model)

predictions_and_scores(model, test_X, test_y)