import os
from pathlib import Path
import sys
#initiate path
sys.path.append('C:/Users/Mariam/Desktop/DataScience/Machine_learning/mlproject/src')
from logger import logging
from exception import CustomException
import pandas as pd

from data_transformation import DataTransformation
from data_transformation import DataTransformationConfig

from model_trainer import ModelTrainer
from model_trainer import ModelTrainerConfig


# import the train test split
from sklearn.model_selection import train_test_split

# import data classes
from dataclasses import dataclass

# DataIngestion Config provides input
@dataclass   # allows us to define the class variable
class DataIngestionConfig:
    # inputs that we give to the data ingestion component and the data ingestion component save the output files to artifact folder
    train_data_path: str = os.path.join('artifact', "train.csv")
    test_data_path: str = os.path.join('artifact', "test.csv")
    raw_data_path: str = os.path.join('artifact', "data.csv")

class DataIngestion:
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # execute DataIngestionConfig Class

    # Start the ingestion, reading the data from somewhere(local path, DB..)
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook/data/stud.csv") # read data from notebook folder
            logging.info("Read the Dataset as Dataframe")
            # getting the directory name(artifact) if it exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            # save data from ingestion to csv file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train Test split initiated")
            # split data onto train set and test set
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # save train and test datasets into different csv files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")
            # return train and test datasets files
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data= obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_tranformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))