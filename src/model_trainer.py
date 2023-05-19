import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from exception import CustomException
from logger import logging
from utils import save_object
from utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    #create the model traning file and join it to artifact folder
    trained_model_file_path=os.path.join("artifact", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()
    #this function initiate the model training by taking parameters returned from data_transformation    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test= (
                train_array[:, :-1],  #all columns except the last col for X_train
                train_array[:,-1], #take just the last col as y_train
                test_array[:,:-1], #all columns except the last col for X_test
                test_array[:,-1], ##the last col for y_test
            )
            models= {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": DecisionTreeRegressor(),
                "Gradien Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "CatBoosting": CatBoostRegressor(),
                "XGB": XGBRegressor(),
                "Adaboost calssifier": AdaBoostRegressor(), 
            }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            #to get the best model score from dict
            best_model_score= max(sorted(model_report.values()))
            
            # to get the best model name
            
            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model= models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model Found")
       
            logging.info(" Best model Found on both training and test datasets")
            
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            # See th predicted best model
            predicted= best_model.predict(X_test)
            r2_square= r2_score(y_test, predicted)
            
            return r2_square
        except CustomException as e:
            
            raise CustomException(e, sys)
        
    