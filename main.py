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

#save data into a csv for testing the confidence interval
data.to_csv('C:\\Users\\geo_m\\PycharmProjects\\Morphemic_TimeSeries\\datasets\\data.csv')

train_X, train_y, test_X, test_y, val_X, val_y = reshape_data_single_lag(data,  0.6, 0.2, 0.2 )


#save data into a csv for testing the confidence interval
train_size = int(len(data) * 0.6)
test_size = int(len(data) * 0.2)
valid_size = int(len(data) * 0.2)

train = data.values[:train_size]
test = data.values[train_size:train_size + test_size]
val = data.values[train_size + test_size:]

X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]
X_val, y_val = val[:, :-1], val[:, -1]

X_train_ = pd.DataFrame(X_train)
X_train_.to_csv('C:\\Users\\geo_m\\PycharmProjects\\Morphemic_TimeSeries\\datasets\\train_X.csv')

y_train_ = pd.DataFrame(y_train)
y_train_.to_csv('C:\\Users\\geo_m\\PycharmProjects\\Morphemic_TimeSeries\\datasets\\train_y.csv')

X_test_ = pd.DataFrame(X_test)
X_test_.to_csv('C:\\Users\\geo_m\\PycharmProjects\\Morphemic_TimeSeries\\datasets\\test_X.csv')

y_test_ = pd.DataFrame(y_test)
y_test_.to_csv('C:\\Users\\geo_m\\PycharmProjects\\Morphemic_TimeSeries\\datasets\\test_y.csv')

X_val_ = pd.DataFrame(X_val)
X_val_.to_csv('C:\\Users\\geo_m\\PycharmProjects\\Morphemic_TimeSeries\\datasets\\val_X.csv')

y_val_ = pd.DataFrame(y_val)
y_val_.to_csv('C:\\Users\\geo_m\\PycharmProjects\\Morphemic_TimeSeries\\datasets\\val_y.csv')

model = LSTM_model(train_X, train_y, test_X, test_y)

model.save('C:\\Users\\geo_m\\PycharmProjects\\Morphemic_TimeSeries\\models_saved\\lstm.h5')

model.summary()

plot_train_test_loss(model)

predictions_and_scores(model, test_X, test_y)