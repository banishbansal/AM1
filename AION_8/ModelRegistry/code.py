
#Standard Library modules
import logging
import sys
import json
import time
import platform
import tempfile
import shutil
import argparse

#Third Party modules
from pathlib import Path

IOFiles = {
    "log": "aion.log",
    "metaData": "modelMetaData.json",
    "model": "model.pkl",
    "performance": "performance.json",
    "production": "production.json",
    "monitor": "monitoring.json"
}
                    
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

        
def validateConfig():        
    config_file = Path(__file__).parent/'config.json'        
    if not Path(config_file).exists():        
        raise ValueError(f'Config file is missing: {config_file}')        
    config = read_json(config_file)		
    return config		

global gl_output_path        
class __register():        
        
    def __init__(self, target_path, usecasename, meta_data, scheduler=True):        
        self.target_path = target_path        
        self.usecasename = usecasename        
        self.meta_data = meta_data        
        self.scheduler = scheduler        
        self.logger = get_logger()        
        self.logger.info('Running Local Registration')        
        
    def setup_registration(self):        
        pass        
        
    def get_unprocessed_runs(self, models):        
        scores = {}        
        self.logger.info('Unprocessed runs:')        
        for model in models:        
            modelperformamnce = model+'_performance.json'        
            performance_file = self.target_path/modelperformamnce        
            if performance_file.exists():        
                with open(performance_file, 'r') as f:        
                    data = json.load(f)        
                scores[model] = data['metrices']['test_score']
                self.logger.info(f"	{model} score: {data['metrices']['test_score']}")
        return scores        
    def get_best_run(self, runs_with_score):        
        best =  max(runs_with_score, key=runs_with_score.get)        
        self.logger.info(f"Best model {best} score: {runs_with_score[best]}")        
        return {'model': best, 'score' : runs_with_score[best]}        
        
    def __copy_to_output(self, source_loc, target_name=None):        
        source = Path(source_loc)        
        if source.is_file():        
            if target_name:        
                target = self.target_path/target_name        
            else:        
                target = self.target_path/(source.name)        
            shutil.copy(source, target)        
            self.logger.info(f'	copying file {source.name} to {target}')        
        
    def __register_model(self,model,score):        
        self.logger.info('Registering Model')        
        meta_data = read_json(self.target_path/IOFiles['metaData'])        
        monitor = read_json(self.target_path/IOFiles['monitor'])        
        production_json = self.target_path/IOFiles['production']        
        production_model = {'Model':model,'runNo':monitor['runNo'],'score':score}        
        write_json(production_model, production_json)        
        
    def is_model_registered(self):        
        return (self.target_path/IOFiles['production']).exists()        

    def __get_registered_model_score(self):        
        data = read_json(self.target_path/IOFiles['production'])        
        self.logger.info(f"Registered Model score: {data['score']}")        
        return data['score']
        
    def __force_register(self,models_info, run):        
        self.__register_model(models_info[run['model']])        
        
    def __scheduler_register(self, models, best_run):        
        if self.is_model_registered():        
            registered_model_score = self.__get_registered_model_score()        
            if registered_model_score >= best_run['score']:        
                self.logger.info('Registered model has better or equal accuracy')        
                return False        
        self.__register_model(best_run['model'],best_run['score'])        
        return True        
        
    def register_model(self,models, best_run):        
        if self.scheduler:        
            self.__scheduler_register(models,best_run)        
        else:        
            self.__force_register(models,best_run)        
        
    def update_unprocessed(self):        
        return None
        
def __merge_logs(log_file_sequence,path, files):        
    if log_file_sequence['first'] in files:        
        with open(path/log_file_sequence['first'], 'r') as f:        
            main_log = f.read()        
        files.remove(log_file_sequence['first'])        
        for file in files:        
            with open(path/file, 'r') as f:        
                main_log = main_log + f.read()        
            (path/file).unlink()        
        with open(path/log_file_sequence['merged'], 'w') as f:        
            f.write(main_log)        
        
def merge_log_files(folder, models):        
    log_file_sequence = {        
        'first': 'aion.log',        
        'merged': 'aion.log'        
    }        
    log_file_suffix = '_aion.log'        
    log_files = [x+log_file_suffix for x in models if (folder/(x+log_file_suffix)).exists()]        
    log_files.append(log_file_sequence['first'])        
    __merge_logs(log_file_sequence, folder, log_files)        
        
def register_model(targetPath,models,usecasename, scheduler, meta_data, ml=False):        
    if ml:        
        raise ValueError('MLFlow Error')        
        register = __ml_register(targetPath, usecasename, meta_data, scheduler)        
    else:        
        register = __register(targetPath,usecasename, meta_data, scheduler)        
    register.setup_registration()        
        
    runs_with_score = register.get_unprocessed_runs(models)        
    best_run = register.get_best_run(runs_with_score)        
    register.register_model(models, best_run)        
        
def register():        
    global gl_output_path        
    config = validateConfig()        
    if platform.system() == 'Windows':        
        targetPath = Path(config['targetPath'])        
    else:        
        targetPath = Path('/aion')/config['targetPath']        
    models = config['models']        
    merge_log_files(targetPath, models)        
    meta_data_file = targetPath/IOFiles['metaData']        
    if meta_data_file.exists():        
        meta_data = read_json(meta_data_file)        
    else:        
        raise ValueError(f'Configuration file not found: {meta_data_file}')        
    usecase = config['targetPath']        
    gl_output_path = targetPath        
    # enable logging        
    log_file = targetPath/IOFiles['log']        
    set_logger(log_file, 'a')        
    scheduler = config.get('scheduler',True)        
    run_id = register_model(targetPath,models,usecase, scheduler, meta_data, ml=False)        
    status = {'Status':'Success','Message':'Model Registered'}        
    get_logger().info(f'output: {status}')        
    return json.dumps(status)
		
if __name__ == '__main__':        
    try:        
        print(register())        
    except Exception as e:        
        if get_logger():        
            get_logger().error(e, exc_info=True)        
        status = {'Status':'Failure','Message':str(e)}        
        print(json.dumps(status))