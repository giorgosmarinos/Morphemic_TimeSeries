import os, json, time, stomp, pickle
from os import path 
from datetime import datetime
from threading import Thread 
from morphemic.dataset import DatasetMaker 

translator_command_queue_name = os.environ.get("TRANSLATOR_QUEUE_NAME","translator_command")
translator_event_queue_name = os.environ.get("TRANSLATOR_QUEUE_NAME","translator_event")
orchestrator_queue_name = os.environ.get("ORCHESTRATOR_QUEUE_NAME","orchestrator")
#/////////////////////////////////////////////////////////////////////////////////
activemq_username = os.getenv("ACTIVEMQ_USER","aaa") 
activemq_password = os.getenv("ACTIVEMQ_PASSWORD","111") 
activemq_hostname = os.getenv("ACTIVEMQ_HOST","localhost")
activemq_port = int(os.getenv("ACTIVEMQ_PORT","61613")) 
#/////////////////////////////////////////////////////////////////////////////////
datasets_path = os.environ.get("DATASET_PATH","./datasets")
ml_model_path = os.environ.get("ML_MODEL_PATH","./models_trained")
prediction_tolerance = os.environ.get("PREDICTION_TOLERANCE","85")
horizons = [60000, 120000, 300000, 600000, 1200000] #in ms [1m, 2m, 5m, 10m, 20m]
#/////////////////////////////////////////////////////////////////////////////////
influxdb_hostname = os.environ.get("INFLUXDB_HOSTNAME","localhost")
influxdb_port = int(os.environ.get("INFLUXDB_PORT","8086"))
influxdb_username = os.environ.get("INFLUXDB_USERNAME","morphemic")
influxdb_password = os.environ.get("INFLUXDB_PASSWORD","password")
influxdb_dbname = os.environ.get("INFLUXDB_DBNAME","morphemic")
influxdb_org = os.environ.get("INFLUXDB_ORG","morphemic")
#//////////////////////////////////////////////////////////////////////////////////

class Predictor():
    def __init__(self):
        pass 

class Listener(object):
    def __init__(self, conn,handler):
        self.conn = conn
        self.handler = handler 

    def on_error(self, headers, message):
        now = datetime.now()
        _time_str = now.strftime("%m/%d/%Y, %H:%M:%S")
        print('received an error {0} at {1}'.format(message, _time_str))

    def on_message(self, headers, message):
        self.handler(message)

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
                conn = stomp.Connection(host_and_ports = [(activemq_hostname, activemq_port)])
                conn.connect(login=activemq_username,passcode=activemq_password)
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
            time.sleep(10)

    def connect(self):
        while True:
            try:
                print('The publisher tries to connect to ActiveMQ broker')
                self.client = stomp.Connection(host_and_ports = [(activemq_hostname, activemq_port)])
                self.client.connect(login=activemq_username,passcode=activemq_password)
                print("connection established")
                return True 
            except:
                pass 

    def send(self):
        if self.message == None or self.queue == None:
            print("Message or queue is None")
            return False 
        try:
            self.client.send(body=json.dumps(self.message), destination=self.message, persistent='false')
            print("Messages pushed to activemq")
            return True 
        except Exception as e:
            print(e)
            self.connect()
            self.send()
            
class Model():
    def __init__(self, application, target, provider, cloud, level, metrics):
        self.application = application
        self.target = target 
        self.provider = provider
        self.cloud = cloud 
        self.level = level 
        self.dataset_creation_time = None 
        self.dataset_url = None 
        self.ml_model_status = 'NotExist'
        self.metrics = metrics 
        self.lowest_prediction_probability = 100

    def getLowestPredictionProbability(self):
        return self.lowest_prediction_probability
    def setLowestPredictionProbability(self,lowest_probability):
        self.lowest_prediction_probability = lowest_probability
    def getMetrics(self):
        return self.metrics
    def setMLModelStatus(self, status):
        self.ml_model_status = status 
    def getMLModelStatus(self):
        return self.ml_model_status
    def setDatasetUrl(self, url):
        self.dataset_url = url 
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
   

class ForecastingManager():
    def __init__(self):
        self.publisher = None 
        self.consumer_manager = None 
        self.applications = {} 

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

    def createModel(self, application, refersTo, provider, cloud, level, metrics):
        if not application in self.applications:
            model = Model(application, refersTo, provider, cloud, level, metrics)
            self.applications[application] = model 
            return True 
        else:
            print('Application already registered')

    def getModel(self,application):
        if application in self.applications:
            return self.applications[application]
        else:
            return None 

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

    def trainModel(self, application, variant):
        if application in self.applications:
            model = self.getModel(application)
            try:
                response = self.prepareDataset(application)
                model.setDatasetUrl(response['url'])
                model.setDatasetCreationTime(time.time())
                #start training ml (application, url, metrics)
                model.setMLModelStatus('started')
            except Exception as e:
                print("An error occured while creating the dataset for the application {0}".format(application))
                print(e)
                return False 

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

    def predict(self,application,model):
        metrics = model.getMetrics()
        predictor = Predictor(application, metrics, horizons)

        results = predictor.predict()
        print(results)
        lowest_probability = 100
        for result in results:
            probability = result['probability']
            horizon = result['horizon']
            value = result['value']
            _time = time.time()
            if result['probability'] < lowest_probability:
                lowest_probability = result['probability']
            self.publish(value,_time,probability,horizon,application,model.getCloud(),model.getProvider(),model.getLevel())
        model.setLowestPredictionProbability(lowest_probability)

    def checkTrainStatus(self, application, model):
        if path.exists(ml_model_path+"/models.obj"):
            models_register = pickle.load(open(ml_model_path+"/models.obj", 'rb'))
            if application in models_register:
                if models_register["application"]["status"] != model.getMLModelStatus():
                    model.setMLModelStatus(models_register["application"]["status"])
                    print("Model for the application {0} changed to status {1}".format(application,models_register["application"]["status"]))

    def run(self):
        print("Forecasting manager started")
        self.publisher = Publisher()
        self.publisher.start()
        list_queues = [translator_command_queue_name,translator_event_queue_name]
        list_handlers = [self.requestDecoder, self.requestDecoder]
        self.consumer_manager = ConsumersManager(list_queues,list_handlers)
        self.consumer_manager.connect()
        while True:
            if list(self.applications.keys()) > 0:
                for application, model in self.applications.items():
                    """
                    We train the model if it doesn't exist or if the prediction precision is very low
                    """
                    if model.getMLModelStatus() == 'NotExist' or model.getLowestPredictionProbability() < prediction_tolerance:
                        self.trainModel(application,"general")
                    elif model.getMLModelStatus() == 'Ready':
                        self.predict(application,model)
                    else:
                        self.checkTrainStatus(application, model)
            time.sleep(60)
                    
        

