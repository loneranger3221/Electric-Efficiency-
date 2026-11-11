import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging


class DataTransformation:
    def __init__(self, house_label="A"):
        self.house_label = house_label.upper()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                f"Total_Electrical_houses_label_{self.house_label}",
                "hour",
                "month",
            ]
            categorical_columns = ["day_of_week", "is_weekend"]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            return ColumnTransformer(
                transformers=[
                    ("num_pipeline", numerical_pipeline, numerical_columns),
                    ("cat_pipeline", categorical_pipeline, categorical_columns),
                ]
            )
        except Exception as error:
            raise CustomException(error, sys) from error

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            target_column = f"Total_Heat_houses_label_{self.house_label}"
            required_columns = {
                "timestamp",
                f"Total_Electrical_houses_label_{self.house_label}",
                "hour",
                "month",
                "day_of_week",
                "is_weekend",
                target_column,
            }
            for name, dataframe in (("train", train_df), ("test", test_df)):
                missing_columns = required_columns.difference(dataframe.columns)
                if missing_columns:
                    raise ValueError(f"{name} data is missing columns: {sorted(missing_columns)}")

            feature_columns = [
                f"Total_Electrical_houses_label_{self.house_label}",
                "hour",
                "month",
                "day_of_week",
                "is_weekend",
            ]
            train_features = train_df[feature_columns]
            test_features = test_df[feature_columns]
            train_target = train_df[target_column].to_numpy()
            test_target = test_df[target_column].to_numpy()
            test_timestamps = test_df["timestamp"].astype(str).to_numpy()

            logging.info("Prepared raw model features for label %s", self.house_label)
            return (
                train_features,
                train_target,
                test_features,
                test_target,
                test_timestamps,
                self.get_data_transformer_object(),
            )
        except Exception as error:
            raise CustomException(error, sys) from error
