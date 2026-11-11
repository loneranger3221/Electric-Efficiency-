# Research Project Report
## Comparative Study of ML/DL Models for Cross-Carrier Energy Load Inference

---

## 1. Dataset

**Title:** Test System for Energy Flexibility
**Source:** IEEE DataPort, DOI: 10.21227/whnz-3v68
**Authors:** Hossein Gharibvand, Gevork B. Gharehpetian, Amjad Anvari-Moghaddam (Aalborg University)
**Released:** December 2025
**Companion paper:** "Energy District Simulations: A Test System for Flexibility Studies," ICTEM 2025, DOI: 10.1109/ICTEM66196.2025.11063529
**File used:** `Electricity_heat_added_2018.xlsx`

**What it contains:** A one-year (2018), hourly-resolution dataset representing a simulated energy district that integrates electricity and heat. It covers three household energy-label categories, electric vehicles, an office building category, and a warehouse category. The file has three sheets: **"Load 2018"** (individual load power values in kW, per physical unit/meter), **"Total loads 2018"** (aggregated electrical and heating loads per category — this is the sheet we use), and **"Usage note"** (column documentation, reproduced in Section 2 below).

**Novelty check performed:** The dataset was released only ~9 months ago (Dec 2025). A search for papers citing the dataset DOI or the companion conference paper found none — no one has yet built an ML/DL model on this data. The source paper itself is a *dataset-construction* paper only; it does not perform any forecasting, prediction, or flexibility quantification on the data it describes. That gap is where this project's contribution sits. (Caveat: paywalled indexes like Scopus/Web of Science were not searchable, so this cannot be treated as an absolute guarantee of novelty — but nothing public exists yet.)

---

## 2. Confirmed Dataset Schema (from the uploaded file)

**Resolution:** 1 hour. **Coverage:** 2018-01-01T00:00:00Z to 2018-12-31T23:00:00Z. **Row count:** 8,760 rows (= 365 days × 24 hours) + header. **Missing values:** none in any column — no imputation is strictly required, though we keep `SimpleImputer` in the pipeline as a safety net.

**Provenance note found in the file:** some of the underlying source data originally came from 2016 and was calendar-shifted onto 2018 by matching weekday-to-weekday (e.g., Saturday to Saturday), with the first 5 days assumed identical across both years since they were holidays. Worth one sentence in the paper's data-description section for transparency.

### `Total loads 2018` sheet — the sheet we use

| Column | Description | Unit | Carrier |
|---|---|---|---|
| `Unnamed: 0` (→ rename `timestamp`) | ISO 8601 UTC timestamp, hourly | — | — |
| `Total_Electrical_houses_label_A` | Total electrical load of 10 houses, energy label A (best-insulated) | kW | Electricity |
| `Total_Electrical_houses_label_B` | Total electrical load of 10 houses, energy label B | kW | Electricity |
| `Total_Electrical_houses_label_D` | Total electrical load of 10 houses, energy label D (least-insulated) | kW | Electricity |
| `Total_EV` | Total EV charging load | kW | Electricity |
| `Total_Heat_houses_label_A` | Total heating load of the same 10 label-A houses | kW | Heat |
| `Total_Heat_houses_label_B` | Total heating load of the same 10 label-B houses | kW | Heat |
| `Total_Heat_houses_label_D` | Total heating load of the same 10 label-D houses | kW | Heat |
| `Total_Electrical_Warehouses` | Total electrical load of 2 warehouses | kW | Electricity |
| `Total_Electrical_office_building` | Total electrical load of 2 office buildings | kW | Electricity |

### Important structural finding — this refines the research idea

**Heat-load data only exists for the three house energy-label categories (A, B, D).** EV, warehouse, and office loads are electricity-only — there is no matching heat column for them in this sheet. This isn't a problem; it actually sharpens the project:

- The cross-carrier inference task (predict heat from electricity) is scoped to houses with labels **A, B, D** — exactly the "three household profiles" the dataset abstract describes.
- Since **energy label A/B/D reflects building insulation quality** (A = best insulated, D = least), comparing inference accuracy across the three labels becomes a genuinely meaningful secondary question: *does the electricity–heat relationship become harder or easier to learn as insulation quality changes?* This is a better-motivated "per-category" comparison than the original placeholder idea (household vs. office vs. warehouse), and it comes for free from the actual data structure.
- EV, warehouse, and office columns are excluded from the target variable, but `Total_EV` and the office/warehouse electrical columns can optionally be kept as extra numeric input features (e.g., testing whether EV charging load in the district adds predictive signal for household heat demand) — optional scope, not required.

---

## 3. Research Idea (finalized)

### The problem framing
Electricity smart meters are cheap and widespread; heat meters are rare and expensive to install in most buildings and districts. This creates a real, recognized need in energy systems research for **"virtual metering" / "soft sensing"** — estimating a quantity that isn't directly measured (heat demand) from a quantity that is (electricity consumption), using calendar/contextual features.

### The research question
**Can a house's heat demand be inferred from its own electricity consumption pattern and calendar features alone, without direct heat metering — and does inference accuracy depend on the building's insulation quality (energy label A vs. B vs. D)?**

### Exact target and feature definition
- **Target (`y`):** `Total_Heat_houses_label_X` (one regression run per label, X ∈ {A, B, D})
- **Primary input feature:** `Total_Electrical_houses_label_X` (same label's electricity load, same timestamp)
- **Calendar features:** `hour` (0–23), `month` (1–12), `day_of_week` (0–6), `is_weekend` (0/1) — all derived from the `timestamp` column
- **Optional extra features (stretch scope only):** `Total_EV`, `Total_Electrical_Warehouses`, `Total_Electrical_office_building` as additional numeric predictors, to test whether district-wide context improves single-house heat inference

### Why this angle, and not the others considered
Three options were evaluated for this dataset:

| Option | Description | Verdict |
|---|---|---|
| A — Cross-carrier inference (chosen) | Predict heat load from electricity load + calendar features at the same timestamp | **Selected** — real-world motivated, minimal template changes, defensible novelty, and the energy-label structure gives a free, meaningful secondary comparison |
| B — Flexibility potential estimation | Predict a custom-defined "flexibility" metric (e.g., shiftable load capacity) | Rejected for this pass — requires inventing and justifying a metric from scratch; more literature work, more risk |
| C — Cross-category transfer learning | Train on one building category, test generalization to another | Rejected as originally scoped — moot anyway, since only houses have heat data to transfer between; the energy-label comparison above captures a cleaner version of this idea for free |

Plain "ML vs. DL load forecasting" (predicting a series' own future from its own past) was explicitly avoided as the core framing, because that exact methodology has been published on other datasets (UK-DALE, REFIT, ASHRAE, ISO-NE, etc.) many times over — using a new dataset alone doesn't make an old method novel. The cross-carrier framing sidesteps this: it's a same-timestamp inference problem (not a future forecast), it's specific to what makes *this* dataset unusual (electricity + heat coupling with an insulation-quality label attached), and it has a stated real-world use case (avoiding the cost of heat metering).

---

## 4. What We Will Actually Do

1. Load the `Total loads 2018` sheet, parse `timestamp`, derive `hour`, `month`, `day_of_week`, `is_weekend`.
2. For each house energy label (A, B, D): build a table of `[Total_Electrical_houses_label_X, hour, month, day_of_week, is_weekend]` → `Total_Heat_houses_label_X`.
3. Train and compare **8 regression models** spanning classical ML, boosting, and a neural network, on the label-A table first (primary case study).
4. Evaluate all 8 models on standard regression metrics plus training/inference time.
5. Repeat steps 3–4 for labels B and D using the exact same code (just re-pointed at different columns), to produce the insulation-quality comparison table.
6. (Stretch, optional) Add `Total_EV`, warehouse, and office electrical columns as extra features and check whether they improve label-A/B/D heat inference.

---

## 5. Models to Train and Compare (8 total)

| # | Model | Family | Source |
|---|---|---|---|
| 1 | Linear Regression | Linear baseline | sklearn (already in template) |
| 2 | KNN Regressor | Instance-based | sklearn (already in template) |
| 3 | Decision Tree Regressor | Tree-based | sklearn (already in template) |
| 4 | Random Forest Regressor | Ensemble — bagging | sklearn (already in template) |
| 5 | AdaBoost Regressor | Ensemble — boosting | sklearn (already in template) |
| 6 | Gradient Boosting Regressor | Ensemble — boosting | sklearn (already in template) |
| 7 | **XGBoost Regressor** | Ensemble — boosting (added) | `pip install xgboost`, same `GridSearchCV` pattern |
| 8 | **MLPRegressor** | Neural network (added) | sklearn (`sklearn.neural_network`) — no PyTorch/GPU setup needed |

**Why 8 and not fewer/more:** Fewer than ~5 looks under-effort for a "comparative study." More than ~10 becomes unwieldy to present and hard to justify each inclusion. 8 spans linear, instance-based, tree, bagging-ensemble, boosting-ensemble, and neural-network families — giving a genuine "families of models" narrative for the methodology section, while every model still slots into the *same* training loop already in the template.

---

## 6. Step-by-Step Pipeline

1. **Data acquisition & structural check** — *(done)* Confirmed hourly resolution, 8,760 rows, no missing values, full 2018 coverage.
2. **Problem formulation** — *(done)* Target and features fixed above.
3. **Feature engineering (minimal)** — Derive `hour`, `month`, `day_of_week`, `is_weekend` from `timestamp` in pandas; no lag features or rolling windows needed since this is same-timestamp inference, not forecasting.
4. **Preprocessing pipeline** — Reuse the template's `ColumnTransformer` almost unchanged: numeric pipeline (impute + `StandardScaler`) for `Total_Electrical_houses_label_X`, `hour`, `month`; categorical pipeline (impute + one-hot + scaler) for `day_of_week` (or treat all as numeric — either is defensible, categorical is closer to the original template's structure).
5. **Train/test split** — Standard random 80/20 split (`train_test_split`, `random_state=42`) — defensible here since there's no future-leakage concern in same-timestamp inference.
6. **Model training** — Run all 8 models through the same `evaluate_models()` / `GridSearchCV` loop already in `model_trainer.py`, with XGBoost and MLPRegressor added to the `models` and `params` dictionaries.
7. **Metrics & timing** — Report R², RMSE, MAE, MAPE per model, plus training time and inference time (via `time.time()` around fit/predict calls).
8. **Per-label repeat** — Re-run the same pipeline for labels A, B, and D using the same code, for the insulation-quality comparison table.
9. **Results write-up** — Comparison table across all 8 models × 3 labels, bar charts of metrics per model, actual-vs-predicted plots for the best model per label, and a discussion of (a) which model family wins and why, (b) accuracy/cost tradeoff, and (c) whether insulation quality (label) affects how learnable the electricity–heat relationship is.

---

## 7. How the Template Maps to This Project — With Real Code

The original template ([`End-to-End-ML-pipeline-`](https://github.com/loneranger3221/End-to-End-ML-pipeline-.git)) originally predicted a student's math score from demographic + exam-score features. Its skeleton — ingestion → transformation → training → prediction — carries over almost entirely intact.

### `src/components/data_ingestion.py`
```python
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

HOUSE_LABEL = 'A'  # change to 'B' or 'D' to re-run for other energy labels

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', f'train_{HOUSE_LABEL}.csv')
    test_data_path: str = os.path.join('artifacts', f'test_{HOUSE_LABEL}.csv')
    raw_data_path: str = os.path.join('artifacts', f'data_{HOUSE_LABEL}.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method')
        try:
            df = pd.read_excel(
                'notebooks/data/Electricity_heat_added_2018.xlsx',
                sheet_name='Total loads 2018'
            )
            df = df.rename(columns={df.columns[0]: 'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

            elec_col = f'Total_Electrical_houses_label_{HOUSE_LABEL}'
            heat_col = f'Total_Heat_houses_label_{HOUSE_LABEL}'
            df = df[[elec_col, 'hour', 'month', 'day_of_week', 'is_weekend', heat_col]]

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info('Ingestion of the data is completed')
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e, sys)
```

### `src/components/data_transformation.py` — only the column lists change
```python
numerical_columns = ['Total_Electrical_houses_label_A', 'hour', 'month']  # rename per label
categorical_columns = ['day_of_week', 'is_weekend']
target_column_name = 'Total_Heat_houses_label_A'  # rename per label
```
Everything else in this file (the `Pipeline`/`ColumnTransformer` construction, `save_object` call) is unchanged from the template.

### `src/components/model_trainer.py` — add 2 models + timing
```python
import time
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

models = {
    'Linear Regression': LinearRegression(),
    'KNN Regressor': KNeighborsRegressor(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    'AdaBoost Regressor': AdaBoostRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(),
    'XGBoost Regressor': XGBRegressor(),                      # added
    'MLP Regressor': MLPRegressor(max_iter=1000),              # added
}

params = {
    'Decision Tree Regressor': {'criterion': ['squared_error', 'friedman_mse', 'absolute_error']},
    'Random Forest Regressor': {'n_estimators': [8, 16, 32, 64, 128, 256]},
    'Gradient Boosting Regressor': {
        'learning_rate': [.05, .1],
        'subsample': [0.8, 1.0],
        'n_estimators': [64, 128]
    },
    'XGBoost Regressor': {                                      # added
        'learning_rate': [.1, .01, .05],
        'n_estimators': [64, 128, 256]
    },
    'MLP Regressor': {                                          # added
        'hidden_layer_sizes': [(64,), (64, 32)],
        'alpha': [0.0001, 0.001]
    },
    'Linear Regression': {},
    'KNN Regressor': {'n_neighbors': [5, 7, 9, 11]},
    'AdaBoost Regressor': {},
}
```

### `src/utils.py` — add timing to `evaluate_models`
```python
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    model_report = {}
    fitted_models = {}
    timing_report = {}
    for name, model in models.items():
        param = params[name]
        grid = GridSearchCV(estimator=model, param_grid=param, cv=3, verbose=1)

        t0 = time.time()
        grid.fit(X_train, y_train)
        train_time = time.time() - t0

        best_model = grid.best_estimator_
        fitted_models[name] = best_model

        t0 = time.time()
        y_test_pred = best_model.predict(X_test)
        inference_time = time.time() - t0

        model_report[name] = r2_score(y_test, y_test_pred)
        timing_report[name] = {'train_time_sec': train_time, 'inference_time_sec': inference_time}

    return model_report, fitted_models, timing_report
```

### Everything else — unchanged
`src/exception.py`, `src/logger.py`, `src/utils.py`'s `save_object`/`load_object`, and the overall `GridSearchCV` pattern stay exactly as in the original template.

### Dropped
- `application.py` (Flask app) — not needed for the paper; drop entirely, or keep as an optional demo only if the guide specifically wants one.

### Filled in (was empty in the original template)
- `src/pipeline/train_pipeline.py` — should become the single orchestrated script that runs ingestion → transformation → training → evaluation for a given `HOUSE_LABEL`, replacing the current `__main__` chaining hack in `data_ingestion.py`.

### Summary of all changes from the original template
- Data source: CSV → Excel (`Total loads 2018` sheet)
- Feature set: demographic/exam scores → electricity load (per house label) + derived calendar features
- Target: math score → heat load (per house label)
- Models: 6 → 8 (added XGBoost, MLPRegressor)
- Metrics: R² only → R², RMSE, MAE, MAPE + training/inference time
- New: `HOUSE_LABEL` parameter to re-run the identical pipeline for labels A, B, D
- `train_pipeline.py`: empty → actual orchestrated entrypoint
- Flask app: kept → dropped (optional demo only)
- Everything else (exception handling, logging, save/load pattern, `ColumnTransformer` structure, `GridSearchCV` pattern): **unchanged**

---

## 8. Summary / Pitch for Your Guide

> We're using the IEEE DataPort "Test System for Energy Flexibility" dataset (released Dec 2025) — a one-year, hourly-resolution, multi-energy-carrier dataset covering electricity and heat loads across households (in three insulation-quality energy labels — A, B, D), EVs, offices, and warehouses in a simulated energy district. No prior work has been built on this dataset yet; the original paper only describes its construction and does not perform any predictive modeling on it.
>
> Our project asks: **can a house's heat demand be inferred from its own electricity consumption pattern and calendar features alone, without direct heat metering — and does this get harder or easier depending on the building's insulation quality?** This is motivated by a real practical gap — electricity meters are common and cheap, heat meters are rare and expensive — making this a form of "virtual metering" relevant to real district energy management. The insulation-quality angle comes directly from the dataset's own structure: it separately tracks electricity and heat for houses with energy labels A, B, and D, letting us test whether the electricity–heat relationship is more or less learnable across insulation classes.
>
> We adapt a standard end-to-end ML pipeline template (data ingestion → preprocessing → training → evaluation) to this problem, and train and compare 8 regression models spanning linear, tree-based, ensemble-boosting, and neural-network families — including Random Forest, Gradient Boosting, XGBoost, and a neural network (MLP) — evaluated on accuracy (R², RMSE, MAE, MAPE) as well as training and inference time, for each of the three house energy labels, to identify (a) which model family offers the best accuracy-versus-computational-cost tradeoff, and (b) whether building insulation quality affects how predictable heat demand is from electricity data alone.

---

*Compiled from project planning discussion and the uploaded dataset file `Electricity_heat_added_2018.xlsx` — [End-to-End-ML-pipeline template](https://github.com/loneranger3221/End-to-End-ML-pipeline-.git) · Dataset DOI: 10.21227/whnz-3v68*

---

## 9. Running the Implemented Pipeline

The training code is configured for the `Total loads 2018` sheet and labels A, B, and D. The dataset itself is not included in this repository. Download the workbook from IEEE DataPort and place it here:

```text
notebooks/data/Electricity_heat_added_2018.xlsx
```

From the project root, install the dependencies and run the complete comparison:

```powershell
python -m pip install -r requirements.txt
python -m src.pipeline.train_pipeline --labels A B D
```

To run one label first, or to use a workbook stored elsewhere:

```powershell
python -m src.pipeline.train_pipeline --labels A
python -m src.pipeline.train_pipeline --labels A B D --data-path "C:\path\to\Electricity_heat_added_2018.xlsx"
```

The script validates the selected sheet, required load columns, and timestamps before splitting the rows (80/20, random seed 42). It trains all eight models and selects the best model separately for each label by the best three-fold cross-validation R² on the training split; test-set metrics are reported separately. Imputation, one-hot encoding, and scaling are fit inside each CV fold to prevent validation-fold preprocessing leakage. The Gradient Boosting search is intentionally bounded to eight parameter combinations to keep the three-label experiment practical. As soon as each model finishes its search and test evaluation, the pipeline writes its metrics and best parameters to the per-label `.txt` log and updates the per-label JSON file. MAPE is expressed as a percentage and excludes rows whose actual heat load is exactly zero; `mape_excluded_zero_count` records the number excluded. Treat the results as a comparison on this dataset rather than an independent final estimate.

Outputs are written under `artifacts`:

| Output | Contents |
|---|---|
| `artifacts/label_A/`, `label_B/`, `label_D/` | Per-label raw, train, and test CSV files |
| `artifacts/label_<X>/model_metrics.csv` | R², RMSE, MAE, MAPE, elapsed training/inference time, best parameters, and selected model |
| `artifacts/label_<X>/predictions.csv` | Actual and predicted test heat load for each model, with timestamps |
| `artifacts/label_<X>/model_training_results.txt` | Human-readable progress log, appended after each model finishes |
| `artifacts/label_<X>/model_results.json` | Valid JSON progress/results document, atomically updated after each model and finalized when the label completes |
| `artifacts/label_<X>/model.pkl` | The best fitted estimator for that label |
| `artifacts/label_<X>/preprocessor.pkl` | The fitted feature preprocessor |
| `artifacts/model_comparison.csv` | Combined model comparison for the labels run |
| `artifacts/model_comparison.json` | Combined, machine-readable JSON comparison (written when all requested labels finish) |

If training is interrupted, the per-label JSON remains parseable and contains the results of models completed so far with `"status": "running"`. Re-run the command to start a fresh run for those labels. The final JSON comparison is written only after every requested label finishes.

To automate a full run from the project root on Windows PowerShell:

```powershell
python -m pip install -r requirements.txt
python -m src.pipeline.train_pipeline --labels A B D
```

For a quick first run, train only label A:

```powershell
python -m src.pipeline.train_pipeline --labels A
```

The JSON progress/results files can be parsed with Python's standard `json` module; the combined JSON has a `labels`, `model_count`, and `results` field. Each result includes model name, cross-validation and held-out metrics, timings, best parameters, and selection status.

### Research interpretation and next steps

This is same-timestamp cross-carrier inference, not future load forecasting. A random hourly split matches that question, but adjacent hourly observations are temporally related; the held-out rows are therefore not an independent future period. State this limitation in the report, and use a chronological holdout as a separate robustness experiment if the intended claim is generalization to later dates. Do not describe the results as proof of performance on another year, district, or unseen building.

After running the pipeline, check that each label has 8 metric rows and that the combined CSV contains the expected labels. Use `predictions.csv` to make the per-model actual-versus-predicted plots and compare the selected models across labels. The workbook’s documented 2016-to-2018 calendar shift should also be disclosed in the data section of the paper. The Flask demo in `application.py` is the original template UI and is not connected to these label-specific research artifacts.
