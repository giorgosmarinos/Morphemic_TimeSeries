
import os, json, time
import pandas as pd 
from datetime import datetime 

dataset_folder = "."
files = ['./EstimatedRemainingTimeContext.csv', './SimulationLeftNumber.csv', './SimulationElapsedTime.csv', './NotFinishedOnTime.csv', './MinimumCoresContext.csv', './NotFinished.csv', './WillFinishTooSoonContext.csv', './NotFinishedOnTimeContext.csv', './MinimumCores.csv', './ETPercentile.csv', './RemainingSimulationTimeMetric.csv', './TotalCores.csv']
tolerance = 1
field_to_keep = ["name","time","value"]
features_list = []

class Row():
    def __init__(self, _time, features):
        self._obj = None 
        try:
            self._obj = datetime.strptime(_time,'%Y-%m-%dT%H:%M:%S.%fZ') 
        except:
            self._obj = datetime.strptime(_time,'%Y-%m-%dT%H:%M:%SZ') 
        self.time = int(self._obj.timestamp())
        self.features = features
    def getTime(self):
        return self.time 
    def addFeatures(self, field_name, filed_value):
        self.features[field_name] = filed_value
    def getFeatures(self):
        return self.features
    def canBeAdded(self, _time):
        return abs(self.time - _time) <= tolerance
    def toRow(self):
        _line = "{0},".format(self.time)
        for key in features_list:
            if key in self.features:
                value = self.features[key]
                _line += "{0}".format(value)+","
            else:
                _line += "null,"
        _line = _line[:-1]
        return  _line 

class Dataset():
    def __init__(self):
        self.rows = {}
        self.size = 0
    def getRow(self,_time):
        for _timestamp, row in self.rows.items():
            if row.canBeAdded(_time):
                return row 
        return None 
    def sortRows(self):
        return sorted(list(self.rows.values()), key=lambda x: x.getTime(), reverse=True)
    def addRow(self, _time, name, value):
        _obj = None 
        try:
            _obj = datetime.strptime(_time,'%Y-%m-%dT%H:%M:%S.%fZ') 
        except:
            _obj = datetime.strptime(_time,'%Y-%m-%dT%H:%M:%SZ') 
        _timestamp = int(_obj.timestamp())
        row = self.getRow(_timestamp)
        if row != None:
            row.addFeatures(name,value)
        else:
            row = Row(_time,{})
            row.addFeatures(name,value)
            self.rows[_timestamp] = row
            self.size +=1 
    def debug(self):
        for k, row in self.rows.items():
            print(k)
            print(row.toRow())
    def build(self):
        content_file = "time,"
        index = 0
        for _time,row in self.rows.items():
            if index == 0:
                #adding title
                for key in features_list:
                    content_file += key+","
                content_file = content_file[:-1]
                content_file+= "\n"

            content_file += row.toRow() + "\n"
            index +=1
        _file = open("dataset.csv","w")
        _file.write(content_file)
        _file.close()

    def getSize(self):
        return self.size

def get_all_files():
    _files = []
    for root, dirs, files in os.walk(dataset_folder):
        for filename in files:
            _files.append(root +  '/' + filename)
    return _files 

def readFiles():
    global features_list
    dataset = Dataset()
    for _file in files:
        df = None 
        try:
            df = pd.read_csv(_file, error_bad_lines=False)
        except:
            continue
        df = df[field_to_keep]
        for row in df.values:
            if not row[0] in features_list:
                features_list.append(row[0])
            try:
                dataset.addRow(row[1],row[0],row[2])
            except Exception as e:
                pass 
    print(dataset.getSize())
    dataset.sortRows()
    dataset.build()
    #dataset.debug()

readFiles()