import pickle 
import sys
import os
from src.exception import CustomException

import dill
'''Dill is a python package which is used to serialize and deserialize python objects. It is an extension of the pickle module and it can serialize a wider range of python objects than pickle.'''

'''Main packages'''
import numpy as np
import pandas as pd

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
 

from sklearn.metrics import r2_score 
from sklearn.model_selection import GridSearchCV
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        model_report = {}
        fitted_models = {}
        
        for name,model in list(models.items()):
            param = params[name]
            
            grid = GridSearchCV(estimator=model, param_grid=param, cv=3, verbose=1)
            grid.fit(X_train, y_train)
            
            best_model = grid.best_estimator_
            fitted_models[name] = best_model
            
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            model_report[name] = test_model_score
            
        return model_report, fitted_models
    except Exception as e:
        raise CustomException(e,sys)
    
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
        