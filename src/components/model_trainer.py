import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelTrainerConfig:
    house_label: str

    @property
    def trained_model_file_path(self):
        return PROJECT_ROOT / "artifacts" / f"label_{self.house_label}" / "model.pkl"

    @property
    def metrics_file_path(self):
        return PROJECT_ROOT / "artifacts" / f"label_{self.house_label}" / "model_metrics.csv"

    @property
    def predictions_file_path(self):
        return PROJECT_ROOT / "artifacts" / f"label_{self.house_label}" / "predictions.csv"

    @property
    def preprocessor_file_path(self):
        return PROJECT_ROOT / "artifacts" / f"label_{self.house_label}" / "preprocessor.pkl"

    @property
    def results_text_file_path(self):
        return PROJECT_ROOT / "artifacts" / f"label_{self.house_label}" / "model_training_results.txt"

    @property
    def results_json_file_path(self):
        return PROJECT_ROOT / "artifacts" / f"label_{self.house_label}" / "model_results.json"


class ModelTrainer:
    def __init__(self, house_label="A"):
        self.house_label = house_label.upper()
        self.model_trainer_config = ModelTrainerConfig(self.house_label)

    def initiate_model_trainer(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        test_timestamps,
        preprocessor,
    ):
        try:
            if len(test_timestamps) != len(y_test):
                raise ValueError("Test timestamps and test targets have different row counts.")

            artifact_dir = self.model_trainer_config.trained_model_file_path.parent
            os.makedirs(artifact_dir, exist_ok=True)
            run_results = {
                "label": self.house_label,
                "status": "running",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "train_rows": len(y_train),
                "test_rows": len(y_test),
                "models": [],
            }
            self._write_json_results(run_results)
            with open(
                self.model_trainer_config.results_text_file_path,
                "w",
                encoding="utf-8",
            ) as results_file:
                results_file.write(
                    f"Model training results for label {self.house_label}\n"
                    f"Started (UTC): {run_results['started_at_utc']}\n"
                    f"Train rows: {len(y_train)} | Test rows: {len(y_test)}\n"
                    "Each model result is written here as soon as its search and evaluation complete.\n"
                )

            def save_completed_model_result(model_name, result):
                result_record = {
                    "model": model_name,
                    **result,
                    "selected": None,
                }
                run_results["models"].append(result_record)
                current_best = max(
                    run_results["models"],
                    key=lambda completed: completed["cv_r2"],
                )
                run_results["current_best_model"] = current_best["model"]
                self._write_json_results(run_results)
                with open(
                    self.model_trainer_config.results_text_file_path,
                    "a",
                    encoding="utf-8",
                ) as results_file:
                    results_file.write(
                        "\n" + json.dumps(result_record, indent=2, allow_nan=False) + "\n"
                    )
                logging.info(
                    "Finished %s for label %s (CV R2=%.4f, test R2=%.4f)",
                    model_name,
                    self.house_label,
                    result["cv_r2"],
                    result["r2"],
                )
                print(
                    f"Label {self.house_label} | {model_name} completed: "
                    f"CV R2={result['cv_r2']:.4f}, test R2={result['r2']:.4f}"
                )

            models = {
                "Linear Regression": LinearRegression(),
                "KNN Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                "Random Forest Regressor": RandomForestRegressor(random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
                "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
                "XGBoost Regressor": XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=1,
                ),
                "MLP Regressor": MLPRegressor(max_iter=1000, random_state=42),
            }
            params = {
                "Linear Regression": {},
                "KNN Regressor": {"n_neighbors": [5, 7, 9, 11]},
                "Decision Tree Regressor": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"]
                },
                "Random Forest Regressor": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "AdaBoost Regressor": {},
                "Gradient Boosting Regressor": {
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.8, 1.0],
                    "n_estimators": [64, 128],
                },
                "XGBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.05],
                    "n_estimators": [64, 128, 256],
                },
                "MLP Regressor": {
                    "hidden_layer_sizes": [(64,), (64, 32)],
                    "alpha": [0.0001, 0.001],
                },
            }

            model_report, fitted_models, predictions = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
                preprocessor=preprocessor,
                on_model_complete=save_completed_model_result,
            )
            best_model_name = max(model_report, key=lambda name: model_report[name]["cv_r2"])
            best_model = fitted_models[best_model_name]
            for result_record in run_results["models"]:
                result_record["selected"] = result_record["model"] == best_model_name

            metrics_rows = []
            prediction_frames = []
            for name, result in model_report.items():
                metrics_rows.append(
                    {
                        "label": self.house_label,
                        "model": name,
                        **{key: value for key, value in result.items() if key != "best_params"},
                        "best_params": json.dumps(result["best_params"], default=str),
                        "selected": name == best_model_name,
                    }
                )
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "timestamp": test_timestamps,
                            "label": self.house_label,
                            "model": name,
                            "actual_heat_kw": y_test,
                            "predicted_heat_kw": predictions[name],
                        }
                    )
                )

            metrics = pd.DataFrame(metrics_rows).sort_values("r2", ascending=False)
            predictions_frame = pd.concat(prediction_frames, ignore_index=True)
            metrics.to_csv(self.model_trainer_config.metrics_file_path, index=False)
            predictions_frame.to_csv(self.model_trainer_config.predictions_file_path, index=False)
            save_object(
                file_path=str(self.model_trainer_config.trained_model_file_path),
                obj=best_model,
            )
            save_object(
                file_path=str(self.model_trainer_config.preprocessor_file_path),
                obj=best_model.named_steps["preprocessor"],
            )

            run_results["status"] = "completed"
            run_results["best_model"] = best_model_name
            run_results["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            self._write_json_results(run_results)
            with open(
                self.model_trainer_config.results_text_file_path,
                "a",
                encoding="utf-8",
            ) as results_file:
                results_file.write(
                    f"\nRun completed (UTC): {run_results['completed_at_utc']}\n"
                    f"Selected model (highest 3-fold CV R2): {best_model_name}\n"
                )

            logging.info(
                "Best model for label %s: %s (3-fold CV R2=%.4f)",
                self.house_label,
                best_model_name,
                model_report[best_model_name]["cv_r2"],
            )
            return metrics, best_model_name
        except Exception as error:
            raise CustomException(error, sys) from error

    def _write_json_results(self, results):
        path = self.model_trainer_config.results_json_file_path
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with open(temporary_path, "w", encoding="utf-8") as results_file:
            json.dump(results, results_file, indent=2, allow_nan=False)
            results_file.write("\n")
        os.replace(temporary_path, path)
