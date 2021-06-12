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
number_of_forward_forecasting = 4
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
   

class ForecastingManager():
    def __init__(self):
        self.publisher = None 
        self.consumer_manager = None 
        self.applications = {} 
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

    def startSendPrediction(self, data):
        global _new_epoch
        _json = json.loads(data)
        metrics = _json["metrics"]
        metrics.append("memory")
        epoch_start = _json["epoch_start"]
        number_forward_predictions = _json["number_of_forward_predictions"]
        prediction_horizon = _json["prediction_horizon"]
        application = "demo"
        sleeping = 0
        _new_epoch = False 
        _now = epoch_start 
        while True:
            _start = time.time()
            index = 1
            while index <= number_forward_predictions:
                horizon = 1*30
                future = horizon * index
                metricValue = random.randint(20,100)
                high_confidence = metricValue + random.randint(2,5)
                low_confidence = metricValue - random.randint(2,5)
                prediction_time = epoch_start + future
                message = {"metricValue": metricValue, "level": 1, "timestamp": int(time.time()), "probability": 0.8,"confidence_interval": [low_confidence,high_confidence], "predictionTime": prediction_time, "refersTo": "demo", "cloud": "aws", "provider": "provider"}
                for metric in metrics:
                    self.publisher.setParameters(message, "intermediate_prediction.cnn.{0}".format(metric))
                    self.publisher.send()
                    print(message, metric)
                index +=1
            sleeping += 30
            time.sleep(horizon)
            epoch_start = prediction_time
                

    def startForecasting(self, data):
        global _new_epoch
        _new_epoch = True 
        print("Start forecasting")
        print(data)
        time.sleep(30)
        self.startSendPrediction(data)
        """
            model = self.getModel(application)
            if model == None:
                print("Model for the application = {0} not found".format(application))
            else:
                #served_request,request_rate,response_time,performance,cpu_usage,memory
                while True:
                    features = {"time":1602543588,"served_request":1906,"request_rate":323,"response_time":645.29249487994,"performance":0.590429925999507,"cpu_usage":26.4,"memory":65351680}
                    self.predict(application, model, metrics, features)
                    time.sleep(10)
        except Exception as e:
            print("Could not decode start forcasting data")
            print(e)
            print(data) """
        #"{"metrics":["AvgResponseTime"],"timestamp":1623242615043,"epoch_start":1623242815041,"number_of_forward_predictions":8,"prediction_horizon":300}
    def simulateForcasting(self):
        application = "default"
        metrics = ["avgResponseTime","memory"] 
        for metric in metrics:
            model = self.getModel(application, metric)
            if model == None:
                print("Model for the application = {0}, metric = {1} not found".format(application, metric))
            else:
                #served_request,request_rate,response_time,performance,cpu_usage,memory
                #features = [2110.0,426.0,673.57,0.63,31.6]
                features = {"time":1602543588,"served_request":1900,"request_rate":300,"avgResponseTime":600,"performance":147,"cpu_usage":20,"memory":65351680}
                response = self.predict(application, model, metric, features)
                print(response)

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
        trainer = Train(application, metric, _time_column_name, model.getDatasetUrl(), number_of_forward_forecasting, steps, model.getPredictionHorizon())
        model.setMLModelStatus('started')
        result = trainer.prepareTraining()
        if len(result) > 0:
            model.setMLModelStatus('Ready')
            model.setTrainingData(result)
            self.saveMorphemicModel()
            self.publishTrainingCompleted(model)
            
    def publish(self, metricValue, timestamp, probability, horizon, application, cloud, provider, level=1):
        """
            {
                "metricValue": 12.34,
                "level": 1,
                "timestamp": 143532341251,
                "probability": 0.98,
                "horizon": 60000,
                "refersTo": "MySQL_12345",
                "cloud": "AWS-Dublin",
                "provider": "AWS"
            }
        """
        message = {'metricValue': metricValue, 'level': level, 'timestamp': timestamp, 'probability': probability,'horizon': horizon}
        message['refersTo'] = application
        message['cloud'] = cloud 
        message['provider'] = provider
        self.publisher.setParameters(message, orchestrator_queue_name)
        self.publisher.send()

    def publishTrainingCompleted(self, model):
        data = model.getTrainingData()
        print(data)
        message = {"metrics": ["cpu_usage"], "forecasting_method":"cnn","timestamp": int(time.time())}
        #print(message)
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
        
        #self.simulateMetricToPredict()
        time.sleep(1)
        self.simulateForcasting()
        while True:
            for key, model in self.applications.items():
                if model.getMLModelStatus() == "NotExist":
                    self.trainModel(model)
            time.sleep(10)
                    
        
if __name__ == "__main__":
    app = ForecastingManager()
    app.run()