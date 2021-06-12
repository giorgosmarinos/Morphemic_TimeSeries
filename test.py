from pre_processing.preprocessing import load_data, percent_missing, datetime_conversion
from pre_processing.preprocessing import important_data, resample, resample_median, missing_data_handling, resample_quantile
from pre_processing.Data_transformation import reshape_data_single_lag, series_to_supervised, \
    prediction_and_score_for_CNN
from models.ML_models import LSTM_model, CNN_model
from plots.plots import plot_train_test_loss
from pre_processing.Data_transformation import predictions_and_scores, Min_max_scal
from pre_processing.Data_transformation import split_sequences
import matplotlib.pyplot as plt
import pandas as pd
import os, time, pickle, json 
from os import path 


#///////////////////////////////////////////////////////////////////////////////
ml_model_path = os.environ.get("ML_MODEL_PATH","./models_trained")
#///////////////////////////////////////////////////////////////////////////////
def load_data2():
    return pd.read_csv("datasets/ds.csv")

metrics = ['performance','request_rate', 'cpu_usage', 'memory','served_request']
metrics = ['cpu_usage', 'memory', 'request_rate',]
data = load_data2()
data = data.round(decimals=2)
data = missing_data_handling(data, rolling_mean=True)
percent_missing(data)
data = datetime_conversion(data, 'time')
print(data)
data = important_data(data, metrics)
print(data)
data = resample(data)
print(data)
data = Min_max_scal(data)
print(data)
#data = series_to_supervised(data, 24, 1)
#print(data)
X_train, y_train, X_test,y_test = split_sequences(data, n_steps=3)
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)
# summarize the data
#for i in range(len(X_train)):
#	print(X_train[i], y_train[i])
#train_X, train_y, test_X, test_y, val_X, val_y = reshape_data_single_lag(data,  0.6, 0.2, 0.2 )
#model = LSTM_model(train_X, train_y, test_X, test_y)
#model.summary()
#plot_train_test_loss(model)
#predictions_and_scores(model, test_X, test_y)

model = CNN_model(n_steps=3, n_features=2, X=X_train, y=y_train, val_x=X_test,  val_y=y_test)
plot_train_test_loss(model)
prediction_and_score_for_CNN(n_steps = 3,n_features=2, x_input=X_test, model=model,test_y=y_test)
model.summary()