from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd


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

    # Create a new model with specified dropout
    if type(model) == Sequential:
        # Sequential
        model_dropout = Sequential.from_config(conf)
    else:
        # Functional
        model_dropout = Model.from_config(conf)
    model_dropout.set_weights(model.get_weights())

    # Create a function to predict with the dropout on
    predict_with_dropout = K.function(model_dropout.inputs + [K.learning_phase()], model_dropout.outputs)

    return predict_with_dropout




dropout = 0.5
num_iter = 20
input_data = pd.read_csv('C:\\Users\\geo_m\\PycharmProjects\\Morphemic_TimeSeries\\datasets\\test_X.csv')
#print(input_data)
num_samples = input_data.values[0].shape[0]

path_to_model = 'C:\\Users\\geo_m\\PycharmProjects\\Morphemic_TimeSeries\\models_saved\\lstm.h5'
model = load_model(path_to_model)

predict_with_dropout = create_dropout_predict_function(model, dropout)

predictions = np.zeros((num_samples, num_iter))
for i in range(num_iter):
    predictions[:,i] = predict_with_dropout(input_data+[1])[0].reshape(-1)


ci = 0.8
lower_lim = np.quantile(predictions, 0.5-ci/2, axis=1)
upper_lim = np.quantile(predictions, 0.5+ci/2, axis=1)