import os, json, time, stomp, pickle
from os import path 
from datetime import datetime
from threading import Thread 
from morphemic.dataset import DatasetMaker 
from main import Train, MorphemicModel, Predictor
import random 
from amq_client.MorphemicConnection import Connection
from amq_client.MorphemicListener import MorphemicListener

translator_command_queue_name = os.environ.get("TRANSLATOR_QUEUE_NAME","topic/translator_command")
translator_event_queue_name = os.environ.get("TRANSLATOR_QUEUE_NAME","/topic/translator_event")
orchestrator_queue_name = os.environ.get("ORCHESTRATOR_QUEUE_NAME","/topic/orchestrator")
#/////////////////////////////////////////////////////////////////////////////////
activemq_username = os.environ.get("ACTIVEMQ_USER","morphemic") 
activemq_password = os.environ.get("ACTIVEMQ_PASSWORD","morphemic") 
activemq_hostname = os.environ.get("ACTIVEMQ_HOST","147.102.17.76")
activemq_port = int(os.environ.get("ACTIVEMQ_PORT","61610")) 
#/////////////////////////////////////////////////////////////////////////////////
datasets_path = os.environ.get("DATASET_PATH","./datasets")
ml_model_path = os.environ.get("ML_MODEL_PATH","./models_trained")
prediction_tolerance = os.environ.get("PREDICTION_TOLERANCE","85")
forecasting_method_name = os.environ.get("FORECASTING_METHOD_NAME","cnn")
#/////////////////////////////////////////////////////////////////////////////////
steps = 128
#/////////////////////////////////////////////////////////////////////////////////
influxdb_hostname = os.environ.get("INFLUXDB_HOSTNAME","localhost")
influxdb_port = int(os.environ.get("INFLUXDB_PORT","8086"))
influxdb_username = os.environ.get("INFLUXDB_USERNAME","morphemic")
influxdb_password = os.environ.get("INFLUXDB_PASSWORD","password")
influxdb_dbname = os.environ.get("INFLUXDB_DBNAME","morphemic")
influxdb_org = os.environ.get("INFLUXDB_ORG","morphemic")
start_forecasting_queue = os.environ.get("START_FORECASTING","/topic/start_forecasting.cnn")
metric_to_predict_queue = os.environ.get("METRIC_TO_PREDICT","/topic/metrics_to_predict")
#//////////////////////////////////////////////////////////////////////////////////
_time_column_name = 'time'
_new_epoch = False 


class Listener(object):
    def __init__(self, conn,handler):
        self.conn = conn
        self.handler = handler 

    def on_error(self, headers, message):
        now = datetime.now()
        _time_str = now.strftime("%m/%d/%Y, %H:%M:%S")
        print('received an error {0} at {1}'.format(message, _time_str))

    def on_message(self, frame):
        self.handler(frame.body)

class Consumer(Thread):
    def __init__(self, handler, queue):
        self.handler = handler 
        self.queue = queue 
        super(Consumer,self).__init__()

    def run(self):
        connected = False 
        while not connected:
            try:
                print('Subscribe to the topic {0}'.format(self.queue))
                conn = Connection(username=activemq_username, password=activemq_password, host=activemq_hostname,port=61610, debug=True)
                conn.connect()
                conn.set_listener('', Listener(conn, self.handler))
                conn.subscribe(destination=self.queue, id=1, ack='auto')
                connected = True 
            except Exception as e:
                print("Could not subscribe")
                print(e)
                connected = False 

class ConsumersManager():
    def __init__(self, list_queues, list_handlers):
        self.list_queues = list_queues
        self.list_handlers = list_handlers

    def connect(self):
        if len(self.list_queues) != len(self.list_handlers):
            print("Cannot proceed, length handler is different to the queue's length")
            raise Exception("Cannot proceed, length handler is different to the queue's length")

        index = 0
        for queue in self.list_queues:
            consumer = Consumer(self.list_handlers[index], queue)
            consumer.start()
            index +=1


class Publisher(Thread):
    def __init__(self):
        self.message = None 
        self.destination = None 
        self.client = None 
        super(Publisher, self).__init__()

    def setParameters(self, message, queue):
        self.message = message
        self.queue = queue 

    def run(self):
        self.connect()
        while True:
            time.sleep(2)

    def connect(self):
        while True:
            try:
                print('The publisher tries to connect to ActiveMQ broker')
                self.client = Connection(username=activemq_username, password=activemq_password, host=activemq_hostname,port=61610, debug=False)
                self.client.connect()
                print("connection established")
                return True 
            except:
                pass 

    def send(self):
        if self.message == None or self.queue == None:
            print("Message or queue is None")
            return False 
        try:
            #self.client.send(body=json.dumps(self.message), destination=self.queue, persistent='false', auto_content_length=False, content_type="application/json")
            self.client.send_to_topic(self.queue, self.message)
            return True 
        except Exception as e:
            print(e)
            self.connect()
            self.send()
   
class Forecaster(Thread):
    def __init__(self, manager, prediction_horizon, epoch_start, publisher, target, application):
        self.manager = manager 
        self.prediction_horizon = prediction_horizon
        self.epoch_start = epoch_start
        self.publisher = publisher
        self.target = target 
        self.application = application
        self.features_dict = {}
        self.stop = False 
        super(Forecaster,self).__init__()

    def getTarget(self):
        return self.target

    def setStop(self):
        self.stop = True 

    def run(self):
        print("Forecaster started for target metric {0} ".format(self.target))
        while True:
            if self.stop:
                print("Forecaster stops after having receiving new epoch start")
                break
            self.features = self.manager.getFeatureInput(self.application)
            if len(self.features) == 0:
                time.sleep(self.prediction_horizon)
                self.epoch_start += self.prediction_horizon
                continue
            predictor = Predictor(self.application, self.target, steps, self.features)
            response = predictor.predict()
            index = 1
            for v, prob,interval in response:
                prediction_time = self.epoch_start + index * self.prediction_horizon
                message = {"metricValue": v, "level": 1, "timestamp": int(time.time()), "probability": prob,"confidence_interval": interval, "predictionTime": prediction_time, "refersTo": self.application, "cloud": "aws", "provider": "provider"}
                #self.publisher.setParameters(message, "intermediate_prediction.cnn.{0}".format(self.target))
                #self.publisher.send()
                print(message, self.target)
                index +=1
            time.sleep(self.prediction_horizon)
            self.epoch_start += self.prediction_horizon

        print("Forecaster for target : {0} stopped".format(self.target))

class ForecastingManager():
    def __init__(self):
        self.publisher = None 
        self.consumer_manager = None 
        self.applications = {} 
        self.forecasting_targets_map = {}
        self.workers = []
        self.loadMorphemicModel()

    def prepareDataset(self,application):
        configs = {'hostname': influxdb_hostname, 
            'port': influxdb_port,
            'username': influxdb_username,
            'password': influxdb_password,
            'dbname': influxdb_dbname,
            'path_dataset': datasets_path
        }
        datasetmaker = DatasetMaker(application,None,configs)
        response = datasetmaker.make()
        return response 

    def createModel(self, application, refersTo, provider, cloud, level, metric, publish_rate):
        if not application+"_"+metric in self.applications:
            model = MorphemicModel(application, refersTo, provider, cloud, level, metric, publish_rate)
            self.applications[application+"_"+metric] = model 
            print("Model created for the application = {0} with metric = {1}".format(application, metric))
            return True 
        else:
            print('Application already registered')

    def getModel(self,application, metric):
        if application+'_'+metric in self.applications:
            return self.applications[application+"_"+metric]
        else:
            return None 

    def getFeatureInput(self, application):
        return [{'time':1602538628,'served_request':2110,'request_rate':426,'avgResponseTime':673.574009325832,'performance':0.626508734240462,'cpu_usage':31.6,'memory':71798784}]
                
    def getModelFromMetric(self, metric):
        for key, model in self.applications.items():
            if model.getMetric() == metric:
                return model 
        return None 

    def startForecasting(self, data):
        _json = None 
        metrics = None 
        epoch_start = None 
        number_of_forward_forecasting = None 
        prediction_horizon = None 
        try:
            _json = json.loads(data)
            metrics = _json['metrics']
            epoch_start = _json['epoch_start']
            number_of_forward_forecasting = _json['number_of_forward_predictions']
            prediction_horizon = _json['prediction_horizon']
            for metric in metrics:
                model = self.getModelFromMetric(metric)
                if model == None:
                    print("Model for metric: {0} does not exist".format(metric))
                    continue
                model.setNumberOfForwardPredictions(number_of_forward_forecasting)
                model.setPredictionHorizon(prediction_horizon)
                model.setEpochStart(epoch_start)
                self.trainModel(model)
            
        except Exception as e:
            print("An error occured in the start forecasting function")
            print(e)

    def simulateForcasting(self):
        data = {"metrics":["avgResponseTime","memory"],"timestamp":1623242615043,"epoch_start":1623242815041,"number_of_forward_predictions":8,"prediction_horizon":30}
        self.startForecasting(json.dumps(data))

    def simulateMetricToPredict(self):
        data = [
            {"refersTo": "default", "level":3, "metric": "avgResponseTime", "publish_rate":3000},
            {"refersTo": "default", "level":3, "metric": "memory", "publish_rate":3000}
        ]
        self.metricToPredict(json.dumps(data))

    def metricToPredict(self, data):
        print("Metric to predict event received")
        try:
            _json = json.loads(data)
            #metrics = ["served_request","request_rate","response_time","performance","cpu_usage","memory"]
            for group in _json:
                application = None 
                if not 'refersTo' in group:
                    application = "demo"
                else:
                    application = group['refersTo']
                self.createModel(application, None, "aws", "cloud", 1, group['metric'], group["publish_rate"])
        except Exception as e:
            print(data)
            print("Could not decode metrics")
            print(e)

    def requestDecoder(self, data):
        try:
            _json = json.loads(data)
            if "request" in _json:
                if _json["request"] == "new_application":
                    application = _json["data"]["application"]
                    metrics = _json["data"]["metrics"]
                    provider = _json["data"]["provider"]
                    level = _json["data"]["level"]
                    cloud = _json["data"]["cloud"]
                    if self.createModel(application,None,provider,cloud,level,metrics):
                        print('Application {0} added successfully'.format(application))

        except Exception as e:
            print("Could not decode JSON content")
            print(e)

    def loadMorphemicModel(self):
        if path.exists(ml_model_path+"/morphemic_models.obj"):
            self.applications = pickle.load(open(ml_model_path+"/morphemic_models.obj", 'rb'))
            print("======Morphemic model found and loaded=======")
            for key, model in self.applications.items():
                print("*******Model*******")
                print("Application: {0}".format(model.getApplication()))
                print("Metric: {0}".format(model.getMetric()))
                print("*******************")
            print("======++++++++++++++++++++++++++++++++=======")

    def saveMorphemicModel(self):
        try:
            pickle.dump(self.applications, open(ml_model_path+"/morphemic_models.obj", 'wb'))
            print("Morhemic Models updated")
        except Exception as e:
            print(e)

    def trainModel(self, model):
        #try:
        application = model.getApplication()
        #response = self.prepareDataset(application)
        #model.setDatasetUrl(response['url'])
        model.setDatasetUrl("/home/jean-didier/Projects/morphemic/Morphemic_TimeSeries/datasets/ds.csv")
        model.setDatasetCreationTime(time.time())
        #start training ml (application, url, metrics)
        metric = model.getMetric()
        trainer = Train(application, metric, _time_column_name, model.getDatasetUrl(), model.getNumberOfForwardPredictions(), steps, model.getPredictionHorizon())
        model.setMLModelStatus('started')
        result = trainer.prepareTraining()
        if len(result) > 0:
            model.setMLModelStatus('Ready')
            model.setTrainingData(result)
            self.saveMorphemicModel()
            self.publishTrainingCompleted(model)
            #start forecaster worker 
            for w in self.workers:
                if w.getTarget() == model.getMetric():
                    w.setStop()
                    time.sleep(5)
            worker = Forecaster(self, model.getPredictionHorizon(), model.getEpochStart(), self.publisher, model.getMetric(), model.getApplication())
            worker.start()
            self.workers.append(worker)

    def publishTrainingCompleted(self, model):
        data = model.getTrainingData()
        message = {"metrics": ["cpu_usage"], "forecasting_method":"cnn","timestamp": int(time.time())}
        print(data)
        #self.publisher.setParameters(message, "training_models")
        #self.publisher.send()

    def predict(self,application,model, target, features):
        predictor = Predictor(application, target, steps, features)
        return predictor.predict()

    def checkTrainStatus(self, application, model):
        if path.exists(ml_model_path+"/morphemic_models.obj"):
            models_register = pickle.load(open(ml_model_path+"/morphemic_models.obj", 'rb'))
            if application in models_register:
                if models_register["application"]["status"] != model.getMLModelStatus():
                    model.setMLModelStatus(models_register["application"]["status"])
                    print("Model for the application {0} changed to status {1}".format(application,models_register["application"]["status"]))

    def run(self):
        print("Forecasting manager started")
        self.publisher = Publisher()
        self.publisher.start()
        list_queues = [translator_command_queue_name,translator_event_queue_name, start_forecasting_queue, metric_to_predict_queue]
        list_handlers = [self.requestDecoder, self.requestDecoder, self.startForecasting, self.metricToPredict]
        self.consumer_manager = ConsumersManager(list_queues,list_handlers)
        self.consumer_manager.connect()
        
        self.simulateMetricToPredict()
        time.sleep(10)
        self.simulateForcasting()
        time.sleep(100)
        self.simulateForcasting()
        while True:
            #for key, model in self.applications.items():
            #    if model.getMLModelStatus() == "NotExist":
            #        self.trainModel(model)
            time.sleep(10)
                    
        
if __name__ == "__main__":
    app = ForecastingManager()
    app.run()