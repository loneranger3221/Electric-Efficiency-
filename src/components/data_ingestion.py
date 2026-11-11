import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_LABELS = ("A", "B", "D")


@dataclass(frozen=True)
class DataIngestionConfig:
    house_label: str
    data_file_path: Path
    artifact_dir: Path

    @property
    def train_data_path(self):
        return self.artifact_dir / f"train_{self.house_label}.csv"

    @property
    def test_data_path(self):
        return self.artifact_dir / f"test_{self.house_label}.csv"

    @property
    def raw_data_path(self):
        return self.artifact_dir / f"data_{self.house_label}.csv"


class DataIngestion:
    def __init__(self, house_label="A", data_file_path=None):
        house_label = house_label.upper()
        if house_label not in SUPPORTED_LABELS:
            raise ValueError(f"house_label must be one of {SUPPORTED_LABELS}")

        source_path = Path(data_file_path) if data_file_path else (
            PROJECT_ROOT / "notebooks" / "data" / "Electricity_heat_added_2018.xlsx"
        )
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path

        self.ingestion_config = DataIngestionConfig(
            house_label=house_label,
            data_file_path=source_path,
            artifact_dir=PROJECT_ROOT / "artifacts" / f"label_{house_label}",
        )

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion for label %s", self.ingestion_config.house_label)
        try:
            if not self.ingestion_config.data_file_path.is_file():
                raise FileNotFoundError(
                    "Dataset not found. Place Electricity_heat_added_2018.xlsx in "
                    "notebooks/data or pass its location with --data-path."
                )

            df = pd.read_excel(
                self.ingestion_config.data_file_path,
                sheet_name="Total loads 2018",
            )
            if df.empty or len(df.columns) < 2:
                raise ValueError("The 'Total loads 2018' sheet is empty or has an invalid schema.")

            df = df.rename(columns={df.columns[0]: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            if df["timestamp"].isna().any():
                raise ValueError("The dataset contains missing or invalid timestamps.")
            if df["timestamp"].duplicated().any():
                raise ValueError("The dataset contains duplicate timestamps.")

            label = self.ingestion_config.house_label
            electricity_column = f"Total_Electrical_houses_label_{label}"
            heat_column = f"Total_Heat_houses_label_{label}"
            required_columns = (electricity_column, heat_column)
            missing_columns = [column for column in required_columns if column not in df.columns]
            if missing_columns:
                raise ValueError(f"Dataset is missing required columns: {missing_columns}")

            df = df.sort_values("timestamp").reset_index(drop=True)
            df["hour"] = df["timestamp"].dt.hour
            df["month"] = df["timestamp"].dt.month
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
            for column in required_columns:
                df[column] = pd.to_numeric(df[column], errors="raise")

            selected_columns = [
                "timestamp",
                electricity_column,
                "hour",
                "month",
                "day_of_week",
                "is_weekend",
                heat_column,
            ]
            df = df[selected_columns]
            os.makedirs(self.ingestion_config.artifact_dir, exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                shuffle=True,
            )
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info(
                "Ingested %s rows for label %s",
                len(df),
                label,
            )
            return str(self.ingestion_config.train_data_path), str(self.ingestion_config.test_data_path)
        except Exception as error:
            raise CustomException(error, sys) from error
