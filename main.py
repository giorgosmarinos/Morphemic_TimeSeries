from pre_processing.preprocessing import load_data, percent_missing, datetime_conversion
from pre_processing.preprocessing import important_data, resample, resample_median, missing_data_handling, resample_quantile
from pre_processing.Data_transformation import reshape_data_single_lag, series_to_supervised, \
    prediction_and_score_for_CNN
from models.ML_models import LSTM_model, CNN_model, CNN_model_multi_steps
from plots.plots import plot_train_test_loss
from pre_processing.Data_transformation import predictions_and_scores, Min_max_scal, Min_max_scal_inverse
from pre_processing.Data_transformation import split_sequences, split_sequences_multi_steps
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import os, time, pickle, json, psutil 
from os import path 
from tensorflow import keras


#///////////////////////////////////////////////////////////////////////////////
ml_model_path = os.environ.get("ML_MODEL_PATH","./models_trained")
ml_model = os.environ.get("ML_MODEL_PATH","./models")
#///////////////////////////////////////////////////////////////////////////////

#metrics = ['performance','request_rate', 'cpu_usage', 'memory','served_request']
"""
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

train_X, train_y, test_X, test_y, val_X, val_y = reshape_data_single_lag(data,  0.6, 0.2, 0.2 )

model = LSTM_model(train_X, train_y, test_X, test_y)

model.summary()

plot_train_test_loss(model)

predictions_and_scores(model, test_X, test_y)

model = CNN_model(n_steps=3, n_features=2, X=X_train, y=y_train, val_x=X_test,  val_y=y_test)
plot_train_test_loss(model)
prediction_and_score_for_CNN(n_steps = 3,n_features=2, x_input=X_test, model=model,test_y=y_test)
model.summary()
"""

class Predictor():
    def __init__(self, application, target, steps, features):
        self.application = application
        self.target = target
        self.steps = steps
        self.feature_dict = features
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

    def makeKey(self, application, target):
        return "{0}_{1}".format(application, target)

    def predict(self):
        key = self.makeKey(self.application, self.target)
        if not key in self.applications_model:
            return {'status': False, 'message': 'Model not found', 'data': None}
        model_metadata = self.applications_model[key]
        path = model_metadata["model_url"]
        print("model path : "+ path)
        #data preparation
        data = self.loadDataset(model_metadata["dataset_url"])
        #if data.empty:
        #    return {'status': False, 'message': 'dataset empty', 'data': None}
        data = data.append(self.feature_dict, ignore_index=True)
        data['memory'] = data['memory']/1000000
        data = data.drop(columns=[self.target, 'time'])
        #data = data.round(decimals=2)
        #data = missing_data_handling(data, rolling_mean=True)
        #percent_missing(data)
        #important_features = model_metadata["features"]
        #important_features.remove(self.target)
        #data = important_data(data, important_features)
        important_feature = list(self.feature_dict.keys())
        important_feature.remove(self.target)
        important_feature.remove('time')
        print(important_feature)
        #data, scaler = Min_max_scal(data)
        print(data)
        data = data.values
        new_sample = data[-self.steps:]
        #new_sample = np.array(self.feature_dict)
        new_sample = new_sample.reshape((1, self.steps, len(important_feature)))

        #new_sample = list()
        #new_sample.append(data)
        #new_sample = np.array(new_sample)
        predictor = keras.models.load_model(path)
        y_predict = predictor.predict(new_sample, verbose=2)
        return y_predict[0].astype('int')
        #y_predict = np.repeat(y_predict, len(important_features)).reshape((-1, len(important_features)))
        #return Min_max_scal_inverse(scaler, y_predict)[-1][-1] #the target is in the last position

class MorphemicModel():
    def __init__(self, application, target, provider, cloud, level, metric, publish_rate):
        self.application = application
        self.target = target 
        self.provider = provider
        self.cloud = cloud 
        self.level = level 
        self.dataset_creation_time = None 
        self.dataset_url = None 
        self.ml_model_status = 'NotExist'
        self.metric = metric
        self.lowest_prediction_probability = 100
        self.publish_rate = publish_rate
        self.prediction_horizon = 30 #30 second 
        self.training_data = None 
        self.features = None 
        self.steps = None 

    def getLowestPredictionProbability(self):
        return self.lowest_prediction_probability
    def setLowestPredictionProbability(self,lowest_probability):
        self.lowest_prediction_probability = lowest_probability
    def getMetric(self):
        return self.metric
    def setMLModelStatus(self, status):
        self.ml_model_status = status 
    def getMLModelStatus(self):
        return self.ml_model_status
    def setDatasetUrl(self, url):
        self.dataset_url = url 
    def getDatasetUrl(self):
        return self.dataset_url
    def setDatasetCreationTime(self,_time):
        self.dataset_creation_time = _time 
    def getDatasetCreationTime(self):
        return self.dataset_creation_time
    def getApplication(self):
        return self.application
    def getTarget(self):
        return self.target 
    def getProvider(self):
        return self.provider
    def getCloud(self):
        return self.cloud 
    def getLevel(self):
        return self.level 
    def getPublishRate(self):
        self.publish_rate
    def getPredictionHorizon(self):
        return self.prediction_horizon
    def setPredictionHorizon(self, prediction_horizon):
        self.prediction_horizon = prediction_horizon
    def getTrainingData(self):
        return self.training_data 
    def setTrainingData(self, training_data):
        self.training_data = training_data 
    def setFeatures(self, features):
        self.features = features
    def getFeatures(self):
        return self.features
    def getsteps(self):
        return self.steps
    def setsteps(self, steps):
        self.steps = steps 

class Model():
    def __init__(self, application, target, steps):
        self.application = application
        self.target = target
        self.steps = steps
        self.status = None 
        self.ml_model = None 
        self.features = None 
        self.training_data = None 
        self.url_dataset = None 
        self.ml_path = None 
        self.dataset_characteristics = None 
        
    def setStatus(self, status):
        self.status = status 
    def getStatus(self):
        return self.status 
    def getMLModel(self):
        return self.ml_model 
    def setMLModel(self, model):
        self.ml_model = model 
    def setMLModelPath(self, path):
        self.ml_path = path 
    def getMLPath(self):
        return self.ml_path
    def setFeatures(self, features):
        self.features = features
    def getFeatures(self):
        return self.features
    def setTrainingData(self,_data):
        self.training_data = _data 
    def getTrainingData(self):
        return self.training_data
    def setUrlDataset(self, url):
        self.url_dataset = url
    def getUrlDataset(self):
        return self.url_dataset
    def getDatasetCharacteristics(self):
        return self.dataset_characteristics
    def setDatasetCharacteristics(self, properties):
        self.dataset_characteristics = properties

class Train():
    def __init__(self, application, metric, _time_column_name, url_dataset, number_of_forward_forecasting, steps, prediction_horizon):
        self.application = application
        self.metric = metric 
        self.features = None 
        self.time_column_name = _time_column_name
        self.applications_model = {} 
        self.url_dataset = url_dataset
        self.number_of_foreward_forecating = number_of_forward_forecasting
        self.steps = steps
        self.prediction_horizon = prediction_horizon
        self.loadModel()

    def loadModel(self):
        if path.exists(ml_model_path+"/models.obj"):
            self.applications_model = pickle.load(open(ml_model_path+"/models.obj", 'rb'))
            print("Application model found and loaded")

    def saveModel(self):
        try:
            pickle.dump(self.applications_model, open(ml_model_path+"/models.obj", 'wb'))
            print("Models updated")
        except Exception as e:
            print(e)

    def loadDataset(self):
        try:
            return pd.read_csv(self.url_dataset, low_memory=False, error_bad_lines=False)
        except Exception as e:
            print("Could not load the dataset")
            print(e)
            return pd.DataFrame()

    def makeKey(self, application, target):
        return "{0}_{1}".format(application, target)

    def canBeTrained(self):
        for key in self.applications_model:
            model = self.applications_model[key]
            if self.application == model['application'] and self.metric == model['target']:
                return False 
        return True 

    def prepareTraining(self):
        result = []
        if not self.canBeTrained():
            print("Model for the metric = {0}, application = {1}, exists".format(self.metric, self.application))
            result.append(self.applications_model[self.makeKey(self.application, self.metric)])
            return result 
        model = Model(self.application, self.metric, self.steps)
        response = self.train(self.metric)
        print(response)
        #////////////////////////////////////////////////
        model.setUrlDataset(self.url_dataset)
        key = None 
        if not response["status"]:
            model.setStatus("Failed")
        else:
            key = self.makeKey(self.application, self.metric)
            model.setStatus("Ready")
            model.setTrainingData(response['training'])
            model.setFeatures(response['training']['features'])
            result.append(response['training'])
            self.applications_model[key] = {"dataset_url":self.url_dataset,"features":response["training"]["features"],"application": self.application, "target": self.metric, "model_url": response['training']['model_url']}
            self.saveModel() 
        return result 

    def computePeriodicity(self, data):
        pass 

    def train(self, target):
        print("Target metric = {0}".format(target))
        print("Sampling rate : {0}".format(self.prediction_horizon))
        data = self.loadDataset()
        data['memory'] = data['memory']/1000000
        if len(data) == 0:
            return {"status": False, "message": "An error occured while loading the dataset", "data": None}

        self.features = list(data.columns.values)
        if not target in self.features:
            return {"status": False, "message": "target not in features list", "data": None}

        if not self.time_column_name in self.features:
            return {"status": False, "message": "time field ({0}) not found in dataset".format(self.time_column_name), "data": None}

        if not self.metric in self.features:
            return {"status": False, "message": "Metric field ({0}) not found in dataset".format(metric), "data": None}
        
        self.features.remove(target)
        self.features.append(target)
        self.features.remove(self.time_column_name)
        ###########
        _start = time.time()
        data = data.round(decimals=2)
        data = missing_data_handling(data, rolling_mean=True)
        print(data)
        percent_missing(data)

        data = datetime_conversion(data, self.time_column_name)
        print(data)
        data = important_data(data, self.features)
        print(data)
        sampling_rate = '{0}S'.format(self.prediction_horizon)
        data = resample_quantile(data, sampling_rate)
        print(data)
        data, scaler = Min_max_scal(data)
        #X_train, y_train, X_test,y_test = split_sequences(data, n_steps=steps)
        X_train, y_train, X_test,y_test = split_sequences_multi_steps(data, n_steps_in=self.steps, n_steps_out=self.number_of_foreward_forecating)
        model = CNN_model_multi_steps(n_steps=self.steps, n_features=len(self.features)-1, X=X_train, y=y_train, val_x=X_test,  val_y=y_test, n_steps_out=self.number_of_foreward_forecating)
        prediction_and_score_for_CNN(n_steps = self.steps,n_features=len(self.features)-1, x_input=X_test, model=model,test_y=y_test)
        _duration = int(time.time() - _start)
        model.summary()
        model_url = ml_model + "/{0}".format(target)
        model.save(model_url)
        process = psutil.Process(os.getpid())
        memory = int(process.memory_info().rss/1000000)
        cpu_usage = psutil.cpu_percent()
        return {"status": True, "message": "success", "training":{"training_duration": _duration,"memory":memory, "cpu_usage": cpu_usage, "algorith": "cnn", "model_url": model_url,"features": self.features, "target": target}}

        
            
    