

from http.server import BaseHTTPRequestHandler,HTTPServer        
from socketserver import ThreadingMixIn        
import os        
from os.path import expanduser        
import platform        
import threading        
import subprocess        
import argparse        
import re        
import cgi        
import json        
import shutil
import logging
import sys        
import time
from pathlib import Path        
from predict import deploy
from groundtruth import groundtruth       

config_input = None        
IOFiles = {
    "inputData": "rawData.dat",
    "metaData": "modelMetaData.json",
	"production": "production.json",
	"log": "aion.log"
}
	
def read_json(file_path):        
    data = None        
    with open(file_path,'r') as f:        
        data = json.load(f)        
    return data        
        
class HTTPRequestHandler(BaseHTTPRequestHandler):        
        
	def do_POST(self):        
		print('PYTHON ######## REQUEST ####### STARTED')        
		if None != re.search('/AION/', self.path) or None != re.search('/aion/', self.path):
			ctype, pdict = cgi.parse_header(self.headers.get('content-type'))        
			if ctype == 'application/json':        
				length = int(self.headers.get('content-length'))        
				data = self.rfile.read(length)        
				usecase = self.path.split('/')[-2]
				if usecase.lower() == config_input['targetPath'].lower():					
					operation = self.path.split('/')[-1]        
					data = json.loads(data)        
					dataStr = json.dumps(data)        
					if operation.lower() == 'predict':        
						output=deployobj.predict(dataStr)        
						resp = output        
					elif operation.lower() == 'groundtruth':
						gtObj = groundtruth(config_input)				
						output = gtObj.actual(dataStr)
						resp = output        
					else:        
						outputStr = json.dumps({'Status':'Error','Msg':'Operation not supported'})        
						resp = outputStr        
				else:
					outputStr = json.dumps({'Status':'Error','Msg':'Wrong URL'})        
					resp = outputStr
        
			else:        
				outputStr = json.dumps({'Status':'ERROR','Msg':'Content-Type Not Present'})        
				resp = outputStr        
			resp=resp+'\n'        
			resp=resp.encode()        
			self.send_response(200)        
			self.send_header('Content-Type', 'application/json')        
			self.end_headers()        
			self.wfile.write(resp)        
		else:        
			print('python ==> else1')        
			self.send_response(403)        
			self.send_header('Content-Type', 'application/json')        
			self.end_headers()        
			print('PYTHON ######## REQUEST ####### ENDED')        
		return        
        
	def do_GET(self):        
		print('PYTHON ######## REQUEST ####### STARTED')        
		if None != re.search('/AION/', self.path) or None != re.search('/aion/', self.path):        
			self.send_response(200)        
			self.send_header('Content-Type', 'application/json')        
			self.end_headers()
			if platform.system() == 'Windows':        
				self.targetPath = Path(config_input['targetPath'])        
			else:        
				self.targetPath = Path('/aion')/config_input['targetPath']			
			meta_data_file = self.targetPath/IOFiles['metaData']        
			if meta_data_file.exists():        
				meta_data = read_json(meta_data_file)        
			else:        
				raise ValueError(f'Configuration file not found: {meta_data_file}')
			features = meta_data['load_data']['selected_features']
			bodydes='['
			for x in features:
				if bodydes != '[':
					bodydes = bodydes+','
				bodydes = bodydes+'{"'+x+'":"value"}'	
			bodydes+=']'
			urltext = 'http://'+config_input['ipAddress']+':'+str(config_input['portNo'])+'/AION/'+config_input['targetPath']+'/predict'
			urltextgth='http://'+config_input['ipAddress']+':'+str(config_input['portNo'])+'/AION/'+config_input['targetPath']+'/groundtruth'
			msg="""
Version:{modelversion}
RunNo: {runNo}
URL for Prediction
==================
URL:{url}
RequestType: POST
Content-Type=application/json
Body: {displaymsg}
Output: prediction,probability(if Applicable),remarks corresponding to each row.

URL for GroundTruth
===================
URL:{urltextgth}
RequestType: POST
Content-Type=application/json
Note: Make Sure that one feature (ID) should be unique in both predict and groundtruth. Otherwise outputdrift will not work  
""".format(modelversion=config_input['modelVersion'],runNo=config_input['deployedRunNo'],url=urltext,urltextgth=urltextgth,displaymsg=bodydes)        
			self.wfile.write(msg.encode())        
		else:        
			self.send_response(403)        
			self.send_header('Content-Type', 'application/json')        
			self.end_headers()        
		return        
        
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):        
	allow_reuse_address = True        
        
	def shutdown(self):        
		self.socket.close()        
		HTTPServer.shutdown(self)        
        
class file_status():
	
	def __init__(self, reload_function, params, file, logger):
		self.files_status = {}
		self.initializeFileStatus(file)
		self.reload_function = reload_function
		self.params = params
		self.logger = logger
		
	def initializeFileStatus(self, file):
		self.files_status = {'path': file, 'time':file.stat().st_mtime}

	def is_file_changed(self):
		if self.files_status['path'].stat().st_mtime > self.files_status['time']:
			self.files_status['time'] = self.files_status['path'].stat().st_mtime
			return True
		return False
		
	def run(self):
		while( True):
			time.sleep(30)
			if self.is_file_changed():
				production_details = targetPath/IOFiles['production']
				if not production_details.exists():        
					raise ValueError(f'Model in production details does not exist')
				productionmodel = read_json(production_details)
				config_file = Path(__file__).parent/'config.json'        
				if not Path(config_file).exists():        
					raise ValueError(f'Config file is missing: {config_file}')        
				config_input = read_json(config_file)				
				config_input['deployedModel'] =  productionmodel['Model']
				config_input['deployedRunNo'] =  productionmodel['runNo']
				self.logger.info('Model changed Reloading.....')
				self.logger.info(f'Model: {config_input["deployedModel"]}')				
				self.logger.info(f'Version: {str(config_input["modelVersion"])}')
				self.logger.info(f'runNo: {str(config_input["deployedRunNo"])}')
				self.reload_function(config_input)
			
class SimpleHttpServer():        
	def __init__(self, ip, port, model_file_path,reload_function,params, logger):        
		self.server = ThreadedHTTPServer((ip,port), HTTPRequestHandler)        
		self.status_checker = file_status( reload_function, params, model_file_path, logger)
		
	def start(self):        
		self.server_thread = threading.Thread(target=self.server.serve_forever)        
		self.server_thread.daemon = True        
		self.server_thread.start()        
		self.status_thread = threading.Thread(target=self.status_checker.run)        
		self.status_thread.start()        
		
	def waitForThread(self):        
		self.server_thread.join()        
		self.status_thread.join()        
		
	def stop(self):        
		self.server.shutdown()        
		self.waitForThread()        
        
if __name__=='__main__':        
	parser = argparse.ArgumentParser(description='HTTP Server')        
	parser.add_argument('-ip','--ipAddress', help='HTTP Server IP')        
	parser.add_argument('-pn','--portNo', type=int, help='Listening port for HTTP Server')        
	args = parser.parse_args()        
	config_file = Path(__file__).parent/'config.json'        
	if not Path(config_file).exists():        
		raise ValueError(f'Config file is missing: {config_file}')        
	config = read_json(config_file)        
	if args.ipAddress:        
		config['ipAddress'] = args.ipAddress        
	if args.portNo:        
		config['portNo'] = args.portNo        
	if platform.system() == 'Windows':		
		targetPath = Path(config['targetPath'])		
	else:		
		targetPath = Path('/aion')/config['targetPath']        
	if not targetPath.exists():        
		raise ValueError(f'targetPath does not exist')
	production_details = targetPath/IOFiles['production']
	if not production_details.exists():        
		raise ValueError(f'Model in production details does not exist')
	productionmodel = read_json(production_details)
	config['deployedModel'] =  productionmodel['Model']
	config['deployedRunNo'] =  productionmodel['runNo']	
	#server = SimpleHttpServer(config['ipAddress'],int(config['portNo']))        
	config_input = config        
	logging.basicConfig(filename= Path(targetPath)/IOFiles['log'], filemode='a', format='%(asctime)s %(name)s- %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')                    
	logger = logging.getLogger(Path(__file__).parent.name)                    
	deployobj = deploy(config_input, logger)
	server = SimpleHttpServer(config['ipAddress'],int(config['portNo']),targetPath/IOFiles['production'],deployobj.initialize,config_input, logger)
	logger.info('HTTP Server Running...........')
	logger.info(f"IP Address: {config['ipAddress']}")
	logger.info(f"Port No.: {config['portNo']}")
	print('HTTP Server Running...........')  
	print('For Prediction')
	print('================')
	print('Request Type: Post')
	print('Content-Type: application/json')	
	print('URL:http://'+config['ipAddress']+':'+str(config['portNo'])+'/AION/'+config['targetPath']+'/predict')	
	print('\nFor GroundTruth')
	print('================')
	print('Request Type: Post')
	print('Content-Type: application/json')	
	print('URL:http://'+config['ipAddress']+':'+str(config['portNo'])+'/AION/'+config['targetPath']+'/groundtruth')	
	print('\nFor Help')
	print('================')
	print('Request Type: Get')
	print('Content-Type: application/json')	
	print('URL:http://'+config['ipAddress']+':'+str(config['portNo'])+'/AION/'+config['targetPath']+'/Help')	
	server.start()        
	server.waitForThread()
