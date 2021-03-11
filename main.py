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

class Predictor():
    def __init__(self, application, target, horizon, features):
        self.application = application
        self.target = target
        self.horizon = horizon
        self.features = features
        self.applications_model = None 
        self.loadModel()

    def loadDataset(self, url):
        try:
            return pd.read_csv(url, low_memory=False, error_bad_lines=False)
        except Exception as e:
            print("Could not load the dataset")
            print(e)
            return None 

    def loadModel(self):
        if path.exists(ml_model_path+"/models.obj"):
            self.applications_model = pickle.load(open(ml_model_path+"/models.obj", 'rb'))
            print("Application model found and loaded")

    def makeKey(self, application, target, horizon):
        return "{0}_{1}_{2}".format(application, target, horizon)

    def predict(self):
        key = self.makeKey(self.application, self.target, self.horizon)
        if not key in self.application_model:
            return {'status': False, 'message': 'Model not found', 'data': None}
        model = self.applications_model[key]
        #data preparation
        data = self.loadDataset(model.getUrlDataset())
        data.loc[0] = list(self.feature.values())
        data = data.round(decimals=2)
        data = missing_data_handling(data, rolling_mean=True)
        percent_missing(data)
        
        data = important_data(data, model.getFeatures())
        data = Min_max_scal(data)
        new_sample = data.iloc[[0]]
        predictor = model.getMLModel()
        x_input = new_sample.drop(self.target)
        yhat = predictor.predict(x_input, verbose=2)


class Model():
    def __init__(self, application, target, horizon):
        self.application = application
        self.target = target
        self.horizon = horizon
        self.status = None 
        self.ml_model = None 
        self.features = None 
        self.training_data = None 
        self.url_dataset = None 
    def setStatus(self, status):
        self.status = status 
    def getStatus(self):
        return self.status 
    def getMLModel(self):
        return self.ml_model 
    def setMLModel(self, model):
        self.ml_model = model 
    def setFeatures(self, features):
        self.features = features
    def getFeatures(self):
        return self.features
    def setTrainingData(self,_data):
        self.training_data = data 
    def getTrainingData(self):
        return self.training_data
    def setUrlDataset(self, url):
        self.url_dataset = url
    def getUrlDataset(self):
        return self.url_dataset

class Train():
    def __init__(self, application, metrics, _time_column_name, url_dataset, horizons):
        self.application = application
        self.metrics = metrics 
        self.features = None 
        self.time_column_name = _time_column_name
        self.applications_model = None 
        self.horizons = horizons
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

    def makeKey(self, application, target, horizon):
        return "{0}_{1}_{2}".format(application, target, horizon)

    def prepareTraining(self):
        for horizon in self.horizons:
            for metric in self.metrics:
                model = Model(self.application, metric, horizon)
                response = self.train(metric)
                #////////////////////////////////////////////////
                model.setUrlDataset(self.url_dataset)
                model.setFeatures(self.metrics)
                if not response["status"]:
                    model.setStatus("Failed")
                else:
                    model.setStatus("Ready")
                    model.setTrainingData(response['training'])
                    model.setMLModel(response['model'])
                key = self.makeKey(self.application, metric, horizon)
                self.applications_model[key] = model 
        self.saveModel()

    def computePeriodicity(self, data):
        pass 

    def train(self, target, horizon=3):
        data = self.loadDataset()
        if data == None:
            return {"status": False, "message": "An error occured while loading the dataset", "data": None}

        self.features = list(data.columns.values)
        if not target in self.features:
            return {"status": False, "message": "target not in features list", "data": None}

        if not self.time_column_name in self.features:
            return {"status": False, "message": "time field ({0}) not found in dataset".format(self.time_column_name), "data": None}

        for metric in self.metrics:
            if not metric in self.features:
                return {"status": False, "message": "Metric field ({0}) not found in dataset".format(metric), "data": None}
        
        #Training 

        data = data.round(decimals=2)
        data = missing_data_handling(data, rolling_mean=True)
        percent_missing(data)
        data = datetime_conversion(data, self.time_column_name)
        #///////////////////////////////////////////////////////////////////////////////////
        index = self.metrics.index(target)
        self.metrics.pop(index)
        self.metrics.append(target) #we need the target as last element of the list metrics
        #///////////////////////////////////////////////////////////////////////////////////
        data = important_data(data, self.metrics)
        data = resample(data)
        data = Min_max_scal(data)
        X_train, y_train, X_test,y_test = split_sequences(data, n_steps=horizon)
        model = CNN_model(n_steps=horizon, n_features=len(self.metrics)-1, X=X_train, y=y_train, val_x=X_test,  val_y=y_test)
        #plot_train_test_loss(model)
        prediction_and_score_for_CNN(n_steps = horizon,n_features=len(self.metrics)-1, x_input=X_test, model=model,test_y=y_test)
        print(model.summary())
        return {"status": True, "message": "", "training": model.summary(), "model": model}

        
            
    