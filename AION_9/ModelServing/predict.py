
#Standard Library modules
import sys
import math
import json
import shutil
import platform
import logging

#Third Party modules
import joblib
import mlflow
import sklearn
import numpy as np 
import pandas as pd 
from pathlib import Path
from influxdb import InfluxDBClient
import category_encoders
from xgboost import XGBClassifier
import word2number as w2n 

IOFiles = {
    "inputData": "rawData.dat",
    "metaData": "modelMetaData.json",
    "performance": "performance.json",
    "prodData": "prodData.dat",
    "log": "predict.log"
}
output_file = { }
                    
def s2n(value):                    
  try:                    
      x=eval(value)                    
      return x                    
  except:                    
      try:                    
          return w2n.word_to_num(value)                    
      except:                    
          return np.nan
                    
def read_json(file_path):                    
    data = None                    
    with open(file_path,'r') as f:                    
        data = json.load(f)                    
    return data                    
                    
def write_json(data, file_path):                    
    with open(file_path,'w') as f:                    
        json.dump(data, f)                    
                    
def read_data(file_path, encoding='utf-8', sep=','):                    
    return pd.read_csv(file_path, encoding=encoding, sep=sep)                    
                    
def write_data(data, file_path, index=False):                    
    return data.to_csv(file_path, index=index)                    
                    
#Uncomment and change below code for google storage                    
#def write_data(data, file_path, index=False):                    
#    file_name= file_path.name                    
#    data.to_csv('output_data.csv')                    
#    storage_client = storage.Client()                    
#    bucket = storage_client.bucket('aion_data')                    
#    bucket.blob('prediction/'+file_name).upload_from_filename('output_data.csv', content_type='text/csv')                    
#    return data                    
                    
def is_file_name_url(file_name):                    
    supported_urls_starts_with = ('gs://','https://','http://')                    
    return file_name.startswith(supported_urls_starts_with)                    

                    
log = None                    
def set_logger(log_file, mode='a'):                    
    global log                    
    logging.basicConfig(filename=log_file, filemode=mode, format='%(asctime)s %(name)s- %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')                    
    log = logging.getLogger(Path(__file__).parent.name)                    
    return log                    
                    
def get_logger():                    
    return log


class database():
    def __init__(self, config):
        self.host = config['host']
        self.port = config['port']
        self.user = config['user']
        self.password = config['password']
        self.database = config['database']
        self.measurement = config.get('measurement', 'measurement')
        self.tags = config['tags']
        self.client = self.get_client()

    def get_client(self):
        client = InfluxDBClient(self.host,self.port,self.user,self.password)
        databases = client.get_list_database()
        databases = [x['name'] for x in databases]
        if self.database not in databases:
            client.create_database(self.database)
        return InfluxDBClient(self.host,self.port,self.user,self.password, self.database)

    def write_data(self,data, tags={}):
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
        for row in data:
            json_body = [{
                'measurement': self.measurement,
                'tags': dict(self.tags, **tags),
                'fields': row
            }]
            res = self.client.write_points(json_body)

    def close(self):
        self.client.close()


class deploy():

    def __init__(self, base_config, logger=None):        
        if platform.system() == 'Windows':        
            self.targetPath = Path(base_config['targetPath'])        
        else:        
            self.targetPath = Path('/aion')/base_config['targetPath']        
        if logger:        
            self.logger = logger        
        else:        
            log_file = self.targetPath/IOFiles['log']        
            self.logger = set_logger(log_file)        
        try:        
            self.initialize(base_config)        
        except Exception as e:        
            self.logger.error(e, exc_info=True)        

    def initialize(self, base_config):        
        self.db_enabled = False        
        self.deployedModel=base_config['deployedModel']+'_model.pkl'        
        self.dataLocation = self.targetPath/IOFiles['inputData']        
        modelmetadata = base_config['deployedModel']+'_metadata.json'        
        meta_data_file = self.targetPath/modelmetadata        
        if meta_data_file.exists():        
            meta_data = read_json(meta_data_file)        
        else:        
            raise ValueError(f'Configuration file not found: {meta_data_file}')        
        self.usecase = base_config['targetPath']        
        self.selected_features = meta_data['load_data']['selected_features']        
        self.train_features = meta_data['training']['features']
        self.missing_values = meta_data['transformation']['fillna']
        self.word2num_features = meta_data['transformation']['word2num_features']
        self.cat_encoder = joblib.load(self.targetPath/meta_data['transformation']['cat_encoder']['file'])
        self.cat_encoder_cols = meta_data['transformation']['cat_encoder']['features']
        self.target_encoder = joblib.load(self.targetPath/meta_data['transformation']['target_encoder'])
        self.model = joblib.load(self.targetPath/self.deployedModel)

    def write_to_db(self, data):
        if self.db_enabled:
            db = database(self.db_config)
            db.write_data(data, {'model_ver': self.model_version[0].version})
            db.close()
        else:
            output_path = self.targetPath/IOFiles['prodData']
            data.to_csv(output_path, mode='a', header=not output_path.exists(), index=False)

    def predict(self, data=None):
        try:
            return self.__predict(data)
        except Exception as e:
            if self.logger:
                self.logger.error(e, exc_info=True)
            raise ValueError(json.dumps({'Status':'Failure', 'Message': str(e)}))

    def __predict(self, data=None):
        if not data:
            data = self.dataLocation
        df = pd.DataFrame()
        if Path(data).exists():
            df=read_data(data,encoding='utf-8')
        elif is_file_name_url(data):
            df = read_data(data,encoding='utf-8')
        else:
            jsonData = json.loads(data)
            df = pd.json_normalize(jsonData)
        if len(df) == 0:
            raise ValueError('No data record found')
        missing_features = [x for x in self.selected_features if x not in df.columns]
        if missing_features:
            raise ValueError(f'some feature/s is/are missing: {missing_features}')
        df_copy = df.copy()
        df = df[self.selected_features]
        for feat in self.word2num_features:
            df[ feat ] = df[feat].apply(lambda x: s2n(x))
        df.fillna(self.missing_values, inplace=True)
        df = self.cat_encoder.transform(df)
        df = df[self.train_features]
        df = df.astype(np.float32)		
        output = pd.DataFrame(self.model.predict_proba(df), columns=self.target_encoder.classes_)        
        df_copy['prediction'] = output.idxmax(axis=1)        
        self.write_to_db(df_copy)        
        df_copy['probability'] = output.max(axis=1).round(2)        
        df_copy['remarks'] = output.apply(lambda x: x.to_json(), axis=1)        
        output = df_copy.to_json(orient='records')
        return output