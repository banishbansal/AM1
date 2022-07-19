#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This file is automatically generated by AION for AION_8_1 usecase.
File generation time: 2022-07-14 13:59:25
'''
#Standard Library modules
import logging
import shutil
import platform
import time
import sys
import json
import argparse

#Third Party modules
from pathlib import Path
import pandas as pd 
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

IOFiles = {
    "inputData": "transformedData.dat",
    "metaData": "modelMetaData.json",
    "log": "aion.log",
    "outputData": "featureEngineeredData.dat"
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

                    
def add_file_for_production(meta_data, file):                    
    if 'prod_files' not in meta_data.keys():                    
        meta_data['prod_files'] = []                    
    if file not in meta_data['prod_files']:                    
        meta_data['prod_files'].append(file)                    
                    
def copy_prod_files(source, target, meta_data):                    
    if 'prod_files' in meta_data.keys():                    
        for file in meta_data['prod_files']:                    
            if not (target/file).exists():                    
                if (source/file).exists():                    
                    shutil.copy(source/file, target/file)
                    
def log_dataframe(df, msg=None):                    
    import io                    
    buffer = io.StringIO()                    
    df.info(buf=buffer)                    
    if msg:                    
        log_text = f'Data frame after {msg}:'                    
    else:                    
        log_text = 'Data frame:'                    
    log_text += '\n\t'+str(df.head(2)).replace('\n','\n\t')                    
    log_text += ('\n\t' + buffer.getvalue().replace('\n','\n\t'))                    
    get_logger().info(log_text)
        
def validateConfig():        
    config_file = Path(__file__).parent/'config.json'        
    if not Path(config_file).exists():        
        raise ValueError(f'Config file is missing: {config_file}')        
    config = read_json(config_file)        
    return config


def featureSelector():        
    config = validateConfig()        
    if platform.system() == 'Windows':		
        targetPath = Path(config['targetPath'])		
    else:		
        targetPath = Path('/aion')/config['targetPath']        
    if not targetPath.exists():        
        raise ValueError(f'targetPath does not exist')        
    meta_data_file = targetPath/IOFiles['metaData']        
    if meta_data_file.exists():        
        meta_data = read_json(meta_data_file)        
    else:        
        raise ValueError(f'Configuration file not found: {meta_data_file}')        
    log_file = targetPath/IOFiles['log']        
    logger = set_logger(log_file)        
    dataLoc = targetPath/IOFiles['inputData']        
    if not dataLoc.exists():        
        return {'Status':'Failure','Message':'Data location does not exists.'}        
        
    status = dict()        
    df = pd.read_csv(dataLoc)        
    prev_step_output = meta_data['transformation']
    train_features = prev_step_output.get('train_features', config['train_features'])
    target_feature = config['target_feature']
    cat_features = config['cat_features']
    total_features = []
    df = df[train_features + [target_feature]]
    log_dataframe(df)
    selected_features = {}
    meta_data['featureengineering']= {}
    logger.info('Model Based Correlation Analysis Start')
    selector = SelectFromModel(ExtraTreesClassifier())
    selector.fit(df[train_features],df[target_feature])
    model_based_feat = df[train_features].columns[(selector.get_support())].tolist()
    if target_feature in model_based_feat:
        model_based_feat.remove(target_feature)
    selected_features['modelBased'] = model_based_feat
    logger.info('Highly Correlated Features : {model_based_feat}')
    total_features = list(set([x for y in selected_features.values() for x in y] + [target_feature]))
    df = df[total_features]
    log_dataframe(df)
                
    csv_path = str(targetPath/IOFiles['outputData'])                
    write_data(df, csv_path,index=False)                
    status = {'Status':'Success','DataFilePath':IOFiles['outputData'],'total_features':total_features, 'selected_features':selected_features}                
    logger.info(f'Selected data saved at {csv_path}')                
    meta_data['featureengineering']['Status'] = status                
    write_json(meta_data, str(targetPath/IOFiles['metaData']))                
    logger.info(f'output: {status}')                
    return json.dumps(status)
        
if __name__ == '__main__':        
    try:        
        print(featureSelector())        
    except Exception as e:        
        if get_logger():        
            get_logger().error(e, exc_info=True)        
        status = {'Status':'Failure','Message':str(e)}        
        print(json.dumps(status))        