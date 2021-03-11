from pre_processing.preprocessing import load_data, percent_missing, datetime_conversion
from pre_processing.preprocessing import important_data, resample, resample_median, missing_data_handling
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

#metrics = ['performance','request_rate', 'cpu_usage', 'memory','served_request']
metrics = ['cpu_usage', 'memory', 'request_rate',]

data = load_data()

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
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# summarize the data
#for i in range(len(X_train)):
#	print(X_train[i], y_train[i])

"""train_X, train_y, test_X, test_y, val_X, val_y = reshape_data_single_lag(data,  0.6, 0.2, 0.2 )

model = LSTM_model(train_X, train_y, test_X, test_y)

model.summary()

plot_train_test_loss(model)

predictions_and_scores(model, test_X, test_y)"""

model = CNN_model(n_steps=3, n_features=2, X=X_train, y=y_train, val_x=X_test,  val_y=y_test)
plot_train_test_loss(model)
prediction_and_score_for_CNN(n_steps = 3,n_features=2, x_input=X_test, model=model,test_y=y_test)
model.summary()

class Train():
    def __init__(self, application, metrics, _time_column_name, url_dataset):
        self.application = application
        self.metrics = metrics 
        self.features = None 
        self.time_column_name = _time_column_name
        self.applications_model = None 
        self.url_dataset = url_dataset
        self.loadModel()

    def loadModel(self):
        if path.exists(ml_model_path+"/models.obj"):
            self.applications_model = pickle.load(open(ml_model_path+"/models.obj", 'rb'))
            print("Application model found and loaded")

    def saveModel(self):
        pickle.dump(self.applications_model, open(ml_model_path+"/models.obj", 'wb'))
        print("Models updated")

    def loadDataset(self):
        try:
            return pd.read_csv(self.url_dataset, low_memory=False, error_bad_lines=False)
        except Exception as e:
            print("Could not load the dataset")
            print(e)
            return None 

    def prepareTraining(self):
        self.applications_model[self.application] = {"status": None}
        for metric in self.metrics:
            self.applications_model[self.application][metric] = {"status": None}

        error = False 
        for metric in self.metrics:
            response = self.train(metric)
            if not response["status"]:
                error = True 
        if error:
            self.applications_model[self.application]["status"] = "Failed"
        else:
            self.applications_model[self.application]["status"] = "Ready"

    def train(self, target):
        error = False 
        data = self.loadDataset()
        if data == None:
            error = True 
            return {"status": False, "message": "An error occured while loading the dataset", "data": None}

        self.features = list(data.columns.values)
        if not target in self.features:
            error = True 
            return {"status": False, "message": "target not in features list", "data": None}

        if not self.time_column_name in self.features:
            error = True 
            return {"status": False, "message": "time field ({0}) not found in dataset".format(self.time_column_name), "data": None}

        for metric in self.metrics:
            if not metric in self.features:
                error = True 
                return {"status": False, "message": "Metric field ({0}) not found in dataset".format(metric), "data": None}

        if error:
            self.applications_model[self.application][target]["status"] = "Failed"
            self.saveModel()
        
        #Training 
        self.applications_model[self.application][target]["status"] = "Training"
        self.saveModel()

        data = data.round(decimals=2)
        data = missing_data_handling(data, rolling_mean=True)
        percent_missing(data)
        data = datetime_conversion(data, self.time_column_name)
        index = self.metrics.index(target)
        self.metrics.pop(index)
        self.metrics.append(target) #we need the target as last element of the list metrics
        data = important_data(data, self.metrics)
        data = resample(data)
        data = Min_max_scal(data)
        X_train, y_train, X_test,y_test = split_sequences(data, n_steps=3)
        model = CNN_model(n_steps=3, n_features=2, X=X_train, y=y_train, val_x=X_test,  val_y=y_test)
        #plot_train_test_loss(model)
        prediction_and_score_for_CNN(n_steps = 3,n_features=2, x_input=X_test, model=model,test_y=y_test)
        print(model.summary())
        self.applications_model[self.application][target]["status"] = "Ready"
        self.applications_model[self.application][target]["model"] = model 
        self.saveModel()
        return {"status": True, "message": "".format(metric), "data": model.summary()}

        
            
    