import argparse
import json
from pathlib import Path

import pandas as pd

from src.components.data_ingestion import DataIngestion, PROJECT_ROOT, SUPPORTED_LABELS
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging


def run_pipeline(labels=SUPPORTED_LABELS, data_path=None):
    labels = tuple(labels)
    if not labels:
        raise ValueError("At least one house label must be selected.")

    results = []
    for label in labels:
        logging.info("Running the research pipeline for label %s", label)
        train_path, test_path = DataIngestion(
            house_label=label,
            data_file_path=data_path,
        ).initiate_data_ingestion()
        (
            X_train,
            y_train,
            X_test,
            y_test,
            test_timestamps,
            preprocessor,
        ) = DataTransformation(house_label=label).initiate_data_transformation(
            train_path, test_path
        )
        metrics, best_model_name = ModelTrainer(house_label=label).initiate_model_trainer(
            X_train,
            y_train,
            X_test,
            y_test,
            test_timestamps,
            preprocessor,
        )
        results.append(metrics)
        print(f"Label {label}: best model is {best_model_name}")

    comparison = pd.concat(results, ignore_index=True)
    results_path = PROJECT_ROOT / "artifacts" / "model_comparison.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(results_path, index=False)
    json_results = comparison.to_dict(orient="records")
    for record in json_results:
        best_params = record.get("best_params")
        if isinstance(best_params, str):
            record["best_params"] = json.loads(best_params)
        for key, value in record.items():
            if key != "best_params" and pd.isna(value):
                record[key] = None
            elif hasattr(value, "item"):
                record[key] = value.item()

    json_results_path = PROJECT_ROOT / "artifacts" / "model_comparison.json"
    temporary_path = json_results_path.with_suffix(".json.tmp")
    with open(temporary_path, "w", encoding="utf-8") as results_file:
        json.dump(
            {
                "labels": list(labels),
                "model_count": len(json_results),
                "results": json_results,
            },
            results_file,
            indent=2,
            allow_nan=False,
        )
        results_file.write("\n")
    temporary_path.replace(json_results_path)
    print(f"Combined model comparison saved to {results_path}")
    print(f"Combined JSON results saved to {json_results_path}")
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare heat-inference models for house labels A, B, and D."
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        choices=SUPPORTED_LABELS,
        default=list(SUPPORTED_LABELS),
        help="House energy labels to run (default: A B D).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Excel dataset path; defaults to notebooks/data/Electricity_heat_added_2018.xlsx.",
    )
    args = parser.parse_args()
    run_pipeline(labels=args.labels, data_path=args.data_path)


if __name__ == "__main__":
    main()
