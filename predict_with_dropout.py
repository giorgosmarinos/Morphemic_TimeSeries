from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import tensorflow as tf
from pre_processing.preprocessing import load_data, percent_missing, datetime_conversion
from pre_processing.preprocessing import important_data, resample, resample_median, missing_data_handling
from pre_processing.Data_transformation import reshape_data_single_lag, series_to_supervised
from models.ML_models import LSTM_model
from plots.plots import plot_train_test_loss, fun_plot_train_test_loss
from pre_processing.Data_transformation import predictions_and_scores, Min_max_scal
import matplotlib.pyplot as plt
import pandas as pd
from models.functional_API_models import fun_LSTM_model

#code inspired by https://medium.com/hal24k-techblog/how-to-generate-neural-network-confidence-intervals-with-keras-e4c0b78ebbdf

def create_dropout_predict_function(model, dropout):
    """
    Create a keras function to predict with dropout
    model : keras model
    dropout : fraction dropout to apply to all layers

    Returns
    predict_with_dropout : keras function for predicting with dropout
    """

    # Load the config of the original model
    conf = model.get_config()
    # Add the specified dropout to all layers
    for layer in conf['layers']:
        # Dropout layers
        if layer["class_name"] == "Dropout":
            layer["config"]["rate"] = dropout
        # Recurrent layers with dropout
        elif "dropout" in layer["config"].keys():
            layer["config"]["dropout"] = dropout
            print(layer)
    # Create a new model with specified dropout
    if type(model) == Sequential:
        # Sequential
        model_dropout = Sequential().from_config(conf)
        print("Seq")
    else:
        # Functional
        model_dropout = Model.from_config(conf)
        print("FUN")
    model_dropout.set_weights(model.get_weights())
    print(model_dropout)
    print(model_dropout.input)
    print(model_dropout.layers)
    print(model_dropout.layers[0])
    print(model_dropout.layers[1].output)
    print(model_dropout.inputs)
    print(model_dropout.outputs)
    # Create a function to predict with the dropout on
    #predict_with_dropout = K.function(model_dropout.inputs + [K.learning_phase()], model_dropout.outputs)
    predict_with_dropout = K.function(model_dropout.inputs + [K.learning_phase()], model_dropout.outputs)

    return predict_with_dropout




dropout = 0.5
num_iter = 20
#input_data = pd.read_csv('C:\\Users\\geo_m\\PycharmProjects\\Morphemic_TimeSeries\\datasets\\train_X.csv')

#code from main.py
metrics = ['performance','request_rate', 'cpu_usage', 'memory','served_request']

data = load_data()

data = data.round(decimals=2)

data = missing_data_handling(data, rolling_mean=True)

percent_missing(data)

data = datetime_conversion(data, 'time')

data = important_data(data, metrics)

data = resample(data)

data = Min_max_scal(data)
#print(data)
data = series_to_supervised(data, 24, 1)
#print(data)
# end of code from main.py

train_X, train_y, test_X, test_y, val_X, val_y = reshape_data_single_lag(data,  0.6, 0.2, 0.2 )
print(train_X.shape)
print(test_X.shape)
print(train_y.shape)
print(test_y.shape)
print(val_y.shape)
print(val_X.shape)


input_data = val_X

num_samples = input_data.shape[0]  #values[0]

path_to_model = 'C:\\Users\\geo_m\\PycharmProjects\\Morphemic_TimeSeries\\models_saved\\lstm.h5'
model = load_model(path_to_model)

predict_with_dropout = create_dropout_predict_function(model, dropout)

predictions = np.zeros((num_samples, num_iter))
for i in range(num_iter):
    predictions[:,i] = predict_with_dropout(input_data+[1])[0].reshape(-1)


ci = 0.8
lower_lim = np.quantile(predictions, 0.5-ci/2, axis=1)
upper_lim = np.quantile(predictions, 0.5+ci/2, axis=1)