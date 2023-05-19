import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer  # to create pipeline
from sklearn.impute import SimpleImputer    #I we have some missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from exception import CustomException  #import exception for exeception Handling
from logger import logging
from utils import save_object

@dataclass  #directly define variables and their data types in a class.
#This give us any path that we may require for the data transformation
class DataTransformationConfig:
    #input file for data transformation
    preprocessor_obj_file_path=os.path.join('artifact', "preprocessor.pkl")
    
class DataTransformation:
        def __init__(self):
            #get the variable defined in datatransformatonconfig
            self.data_transformation_config= DataTransformationConfig()
          
        # function to transform data, from cat to num, scaling features, preprocessing
        def get_data_transformer_object(self):
            try:
                numerical_columns= ["writing_score", "reading_score"]
                categorical_columns= [
                    "gender",
                    "race_ethnicity",
                    "parental_level_of_education",
                    "lunch",
                    "test_preparation_course"
                ]
                #perform pipeline on numerical training dataset
                num_pipeline= Pipeline(
                    steps= [
                        ("imputer", SimpleImputer(strategy="median")),  #handle missing values with median
                        ("scaler", StandardScaler(with_mean=False))  # feature scaling
                    ]
                )
                #perform pipeline on catgeorical training features
                cat_pipeline= Pipeline(
                    steps= [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("One_hot_encoder", OneHotEncoder()),
                        ("scaler", StandardScaler(with_mean=False))
                    ]
                )
                
                logging.info("Numerical Columns scaling Completed")
                logging.info("Catgorical Columns Encoding Completed")
                
                # Combine categorical and Numerical pipelines
                preprocessor= ColumnTransformer(
                    [
                        ("num_pipeline", num_pipeline, numerical_columns),
                        ("cat_pipeline", cat_pipeline, categorical_columns)
                    ]
                )
                return preprocessor
            except Exception as e:
                raise CustomException(e, sys)
        def initiate_data_tranformation(self, train_path, test_path):
            try:
                train_df= pd.read_csv(train_path)
                test_df= pd.read_csv(test_path)
                logging.info("Read train test dataset completed")
                logging.info("Obtaining preprocessed object")
                
                # Read all preprocessor object
                preprocessing_obj= self.get_data_transformer_object()
                #train_df['total_score'] = train_df['math_score'] + train_df['reading_score'] + train_df['writing_score']
                #train_df['average'] = train_df['total_score']/3
                #target_column_name= train_df["average"]
                target_column_name= "math_score"
                numerical_columns= "writing_score", "reading_score"
                #specify input and output variables on train ds
                input_feature_train_df= train_df.drop(columns=[target_column_name], axis=1) 
                target_feature_train_df= train_df[target_column_name]
                
                 #specify input and output variables on test ds
                input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
                target_feature_test_df=test_df[target_column_name]
                
                logging.info(f" Applying preprocessing object on training and test dataframes")
                
                input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)
                
                #convert train and test arr into numpy array
                train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
                logging.info(f"Saved preprocessing object.")
                
                # To save filescls
                save_object(
                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj= preprocessing_obj
                )
                
                return(
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path
                )
            except Exception as e:
                raise CustomException(e, sys)
                
                