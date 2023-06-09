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
                "Random Forest": RandomForestRegressor(max_depth=2, random_state=0),
                "Gradient Boosting": DecisionTreeRegressor(random_state=42),
                "Gradien Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(n_neighbors=2),
                "CatBoosting": CatBoostRegressor(depth=10),
                "XGB": XGBRegressor(n_estimators=100),
                "Adaboost calssifier": AdaBoostRegressor(random_state=0, n_estimators=100), 
            }
            #Hyper paramter Tuning
            '''
            params= {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            '''
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
            # See the predicted best model
            predicted= best_model.predict(X_test)
            r2_square= r2_score(y_test, predicted)
            
            return r2_square
        except CustomException as e:
            
            raise CustomException(e, sys)
        
    