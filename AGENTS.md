# AGENTS.md

## Project Overview

Bicycle booking prediction for the BIXI Montreal bike-sharing system. Predicts trip demand from Metro Mont-Royal (station 6184) to Berri/de Maisonneuve (station 6015) using historical ride data from 2014-2017 (April-November seasons).

Two prediction granularities:
- **Daily classification** (XGBClassifier) with 4 demand bins (low/mid/high/peak) and SMOTE balancing, achieving 97.6% test accuracy. Regression models performed poorly due to low daily trip counts.
- **3-hour bin classification** using station neighbor groups (K=5 nearest stations per target). 8 bins per day (0:00, 3:00, ..., 21:00). Temporal split: 2014-2016 train, 2017 test. No holiday features in this pipeline.
  - *Single-stage (notebook 08)*: 4-class GradientBoosting — 77% accuracy.
  - *Two-stage (notebook 09)*: Stage 1 binary (zero vs non-zero, 94% acc) → Stage 2 binary (low 1-3 trips vs high 4+, 88% acc). **Combined 3-class accuracy: 89%, weighted F1: 0.88.** Best hourly model.

## Repository Structure

```
Bicycles-Booking-Prediction/
├── Data/                    # Consolidated CSVs: OD_YYYY-MM.csv (rides), Stations_YYYY.csv
├── Downloaded_data/         # Raw downloaded BIXI data archives (2014-2017)
├── Data_preparation/        # Jupyter notebooks for data pipeline
│   ├── 01_Pre_analysis.ipynb
│   ├── 02_Data_aggregation.ipynb
│   ├── 03_Feature_engineering.ipynb
│   ├── 04_Data_balance.ipynb
│   ├── 05_Data_bins.ipynb
│   ├── 06_Station_neighbors.ipynb
│   ├── 07_Hourly_aggregation.ipynb
│   ├── 08_Hourly_features_and_model.ipynb
│   └── 09_Hourly_two_stage_model.ipynb
├── Project_datasets/        # Processed datasets (CSVs in .gitignore)
├── Project_goal             # Project goal description (text file)
├── Train_model/
│   └── 01_Model_selection.ipynb
└── AGENTS.md
```

## Data Pipeline

Notebooks must be run in order, as each depends on outputs from the previous step.

### Daily pipeline (notebooks 01-05)

1. **01_Pre_analysis.ipynb** - Explores raw data structure, identifies target stations and verifies consistency across years.
2. **02_Data_aggregation.ipynb** - Merges 2014-2017 ride data, creates unified station codes, aggregates rides by date/route. Outputs `Aggregated_all_rides_codes.csv`.
3. **03_Feature_engineering.ipynb** - Filters to the target route and creates 25 features: calendar (day_of_week, is_weekend, month, season, is_holiday), lag (1/2/7/14/month/year), rolling window (mean/std/max), and system-wide demand indicators. Outputs `Features.csv`.
4. **04_Data_balance.ipynb** - Bins trip counts into 4 classes (low/mid/high/peak), applies SMOTE to balance classes. Outputs `cleaned_bins.csv` (254/class) and `cleaned_bins_big.csv` (1000/class).
5. **05_Data_bins.ipynb** - Validates regression vs classification approach, trains XGBClassifier on balanced data.

### Hourly pipeline (notebooks 06-09, depends on 02)

6. **06_Station_neighbors.ipynb** - Finds K=5 nearest stations to each target station using Haversine distance. Creates station groups (6 stations each: target + 5 neighbors). Outputs `station_groups.csv` and `station_code_lookup.csv`.
7. **07_Hourly_aggregation.ipynb** - Aggregates rides into 3-hour bins for rides starting in the Mont-Royal group and ending in the Berri group. Fills missing bins with zeros. Outputs `hourly_grouped_rides.csv`.
8. **08_Hourly_features_and_model.ipynb** - Creates 3-hour bin features (calendar, cyclical encoding, rush period flags, lags at 1/2/3/8/56 bin offsets, rolling windows, interaction terms). No holiday features. Trains single-stage 4-class classification (RandomForest, GradientBoosting, XGBoost). Saves `hourly_features.csv`. Train/test split: 2014-2016 / 2017.
9. **09_Hourly_two_stage_model.ipynb** - Two-stage 3-class classification. Loads `hourly_features.csv` from notebook 08. Stage 1: GradientBoosting binary (zero vs non-zero). Stage 2: GradientBoosting binary (low 1-3 vs high 4+) on non-zero subset. Both stages use SMOTE. Includes cross-validation (TimeSeriesSplit, 5 folds) and per-stage feature importance. Outputs `model_results_hourly_two_stage.csv`.

## Model Training

**01_Model_selection.ipynb** - Compares regression (XGBoost, Random Forest, Poisson) and classification (XGBClassifier) for daily prediction. Classification with SMOTE-balanced bins is the recommended daily approach.

**08_Hourly_features_and_model.ipynb** - Single-stage 4-class hourly classification (77% accuracy).

**09_Hourly_two_stage_model.ipynb** - Two-stage 3-class hourly classification (89% accuracy, best hourly model).

## Key Datasets (git-ignored)

| File | Description |
|------|-------------|
| `Aggregated_all_rides_codes.csv` | All rides with unified station codes (~15.5M rows) |
| `Features.csv` | Daily features for target route (718 rows x 25 features) |
| `cleaned_bins.csv` | SMOTE-balanced daily classification (254 samples/class) |
| `cleaned_bins_big.csv` | SMOTE-balanced daily classification (1000 samples/class) |
| `station_groups.csv` | Target + neighbor stations (12 rows) |
| `station_code_lookup.csv` | Year-to-code mapping for station groups |
| `hourly_grouped_rides.csv` | 3-hour bin aggregated rides for station groups |
| `hourly_features.csv` | 3-hour bin features (36 cols, saved by notebook 08) |
| `model_results_hourly_two_stage.csv` | Two-stage model results summary |

## Key Libraries

- `pandas`, `numpy` - data manipulation
- `xgboost` - XGBClassifier / XGBRegressor
- `scikit-learn` - RandomForest, GradientBoosting, PoissonRegressor, train/test split, metrics
- `imblearn` - SMOTE oversampling
- `holidays` - Canadian holiday calendar (daily pipeline only)
- `matplotlib`, `seaborn` - visualization

## Coding Conventions

- Notebooks are numbered sequentially within each folder to indicate execution order.
- Dataset CSVs in `Project_datasets/` are git-ignored; regenerate by running the data preparation pipeline.
- Station codes are unified across years (original codes vary by year).
- No `requirements.txt` exists; dependencies must be inferred from notebook imports.

## Common Tasks

- **Regenerate daily datasets**: Run notebooks 01-05 in `Data_preparation/` sequentially.
- **Regenerate hourly datasets**: Run notebook 02 (if not already done), then 06-08 sequentially.
- **Retrain daily model**: Run `Train_model/01_Model_selection.ipynb` after daily datasets are generated.
- **Retrain hourly model**: Run notebooks 08 then 09 in `Data_preparation/` after hourly datasets are generated.
- **Add new year of data**: Update `02_Data_aggregation.ipynb` to include new CSVs in `Data/`, then re-run the relevant pipeline(s).
