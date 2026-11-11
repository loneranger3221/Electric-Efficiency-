import os
import sys
from src.exception import CustomException
from src.logger import logging

'''main data ingestion packages '''
import pandas as pd
from sklearn.model_selection import train_test_split

'''packages from other components'''
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

'''dataclasses is a decorator that automatically generates special methods like __init__() and __repr__() for classes, making it easier to create classes that primarily store data.'''
from dataclasses import dataclass

'''In data ingestion we have some configuration related to data ingestion, so we will create a dataclass for that'''

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')
    '''os.path.join is used to create a file path by joining one or more path components. It ensures that the correct path separator is used based on the operating system. In this case, it creates a path for the train and test data files in the 'artifacts' directory.'''

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df=pd.read_csv(os.path.join('notebooks/data','StudentsPerformance.csv'))
            logging.info('Read the dataset as dataframe')
            
            '''Now we will create the directory to store the data if it does not exist'''
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('train test split initiated')
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info('Ingestion of the data is completed')
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    
    model_trainer=ModelTrainer()
    print('Best model score is : {}'.format(model_trainer.initiate_model_trainer(train_arr,test_arr)))
    