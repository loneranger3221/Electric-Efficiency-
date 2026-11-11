import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass 
from src.utils import save_object, evaluate_models

'''Main packages for model training '''

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                'Linear Regression': LinearRegression(),
                'KNN Regressor': KNeighborsRegressor(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor()
            }
            '''Hyperparameters tuning for models'''
            
            params= {
                'Decision Tree Regressor': {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                'Random Forest Regressor': {
                    'n_estimators': [8,16,32,64,128,256]
                },
                'Gradient Boosting Regressor': {
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'Linear Regression': {},
                'KNN Regressor': {
                    'n_neighbors': [5,7,9,11],
                    },
                'AdaBoost Regressor':{}
            }
            
            
            model_report, fitted_models = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
            
            best_model_score = max(model_report.values())
            best_model_name = [k for k, v in model_report.items() if v == best_model_score][0]
            
            best_model = fitted_models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException('No best model found with R2 score greater than 0.6', sys)
            
            logging.info(f'Best model found on both training and testing dataset: {best_model_name} with R2 score: {best_model_score}')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            return best_model_score
            
        except Exception as e:
            logging.info('Exception occurred during model training')
            raise CustomException(e, sys)
