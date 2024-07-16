import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model_path = 'artifacts/model.pkl'
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 footfall:int,
                 tempMode:int,
                 AQ:int,
                 USS:int,
                 CS:int,
                 VOC:int,
                 RP:int,
                 IP:int,
                 Temperature:int):
        
        self.footfall = footfall
        self.tempMode = tempMode
        self.AQ = AQ
        self.USS = USS
        self.CS = CS
        self.VOC = VOC
        self.RP = RP
        self.IP = IP
        self.Temperature = Temperature

    def get_daRPta_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'footfall':[self.footfall],
                'tempMode':[self.tempMode],
                'AQ':[self.AQ],
                'USS':[self.USS],
                'CS':[self.CS],
                'VOC':[self.VOC],
                'RP':[self.RP],
                'IP':[self.IP],
                'Temperature':[self.Temperature]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
            