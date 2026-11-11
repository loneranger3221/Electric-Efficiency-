import os
import sys
from time import perf_counter

import dill
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    params,
    preprocessor,
    on_model_complete=None,
):
    try:
        model_report = {}
        fitted_models = {}
        predictions = {}

        for name, model in models.items():
            model_pipeline = Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("regressor", model),
                ]
            )
            param = {
                f"regressor__{key}": values
                for key, values in params[name].items()
            }

            grid = GridSearchCV(
                estimator=model_pipeline,
                param_grid=param,
                cv=3,
                scoring="r2",
                n_jobs=1,
                error_score="raise",
            )
            train_start = perf_counter()
            grid.fit(X_train, y_train)
            train_time = perf_counter() - train_start

            best_model = grid.best_estimator_
            fitted_models[name] = best_model

            inference_start = perf_counter()
            y_test_pred = np.asarray(best_model.predict(X_test), dtype=float)
            inference_time = perf_counter() - inference_start
            predictions[name] = y_test_pred

            y_test_array = np.asarray(y_test, dtype=float)
            nonzero_targets = y_test_array != 0
            mape_pct = (
                float(
                    np.mean(
                        np.abs(
                            (y_test_array[nonzero_targets] - y_test_pred[nonzero_targets])
                            / y_test_array[nonzero_targets]
                        )
                    )
                    * 100
                )
                if np.any(nonzero_targets)
                else None
            )
            model_report[name] = {
                "cv_r2": float(grid.best_score_),
                "r2": float(r2_score(y_test_array, y_test_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test_array, y_test_pred))),
                "mae": float(mean_absolute_error(y_test_array, y_test_pred)),
                "mape_pct": mape_pct,
                "mape_excluded_zero_count": int(np.count_nonzero(~nonzero_targets)),
                "train_time_sec": float(train_time),
                "inference_time_sec": float(inference_time),
                "best_params": {
                    key.split("__", 1)[-1]: value
                    for key, value in grid.best_params_.items()
                },
            }
            if on_model_complete is not None:
                on_model_complete(name, model_report[name])

        return model_report, fitted_models, predictions
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)