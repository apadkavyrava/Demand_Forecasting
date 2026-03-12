# Model Evaluation and Experimentation Report

## Problem

Predict bicycle trip demand on the route from X station to Y station using 2014-2017 historical data. 

---

## Phase 1: Daily Regression

**Approach:** Predict the exact number of daily trips using regression models on 535 data points (80/20 random split).

**Features (23):** Calendar (day_of_week, is_weekend, month, season, is_holiday), lag features (1/2/7/14-day, monthly, yearly), rolling windows (3-day mean, 7-day std/max), and system-wide demand indicators (total trips, trips from Mont-Royal, trips to Berri).

| Model | R² | RMSE |
|-------|-----|------|
| XGBRegressor | -0.144 | 2.54 |
| RandomForestRegressor | -0.027 | 2.40 |
| PoissonRegressor | -0.000 | 2.37 |
| Baseline (mean) | 0.000 | — |

**Conclusion:** All models at or below baseline. Daily trip counts are too low (1-7 trips/day) for regression. Switched to classification.

---

## Phase 2: Daily Classification

**Approach:** Bin daily trip counts into 4 demand classes and train on SMOTE-balanced data (1000 samples/class, up from 32-254 original). 80/20 random split.

**Bins:** low (1-2), mid (3-4), high (5-6), peak (7+)

*An exaple severe overfitting when the dataset was increase to 10000 samples from 535 original samples.*
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| low (1-2) | 0.96 | 0.96 | 0.96 | 191 |
| mid (3-4) | 0.97 | 0.97 | 0.97 | 197 |
| high (5-6) | 0.98 | 0.98 | 0.98 | 205 |
| peak (7+) | 1.00 | 1.00 | 1.00 | 207 |
| **Overall** | | | **0.98** | **800** |

- *Train Accuracy: 100%*
- *Test Accuracy: 97.6%*
  
**Conclusion:** Data generation did not improve the results; no threshold was found where the model was neither overfitting nor producing poor predictions.

---

## Phase 3: Hourly (3-Hour Bin) Classification — Single Stage

**Approach:** 
- Increase granularity to 3-hour bins (8 bins/day).
- Expanded station scope using K=5 nearest neighbors per target station.

**Increase number of features (36):** Calendar, cyclical encoding (sin/cos for hour, day-of-week, month), rush period flags, lags at 1/2/3/8/56 bin offsets, rolling windows, interaction features, time period dummies.

**Target:** 4 classes — zero (0 trips), low (1), mid (2-3), high (4+)

### Model Comparison

| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| RandomForest | 73.6% | 0.724 |
| **GradientBoosting** | **77.0%** | **0.760** |
| XGBoost | 76.3% | 0.753 |


### Per-Class Results — GradientBoosting (77.0%)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| zero | 0.92 | 0.99 | 0.95 | 985 |
| low | 0.58 | 0.43 | 0.49 | 339 |
| mid | 0.56 | 0.63 | 0.59 | 397 |
| high | 0.65 | 0.49 | 0.56 | 175 |


### Cross-Validation (5-fold TimeSeriesSplit, XGBoost)

| Fold | Accuracy | Weighted F1 |
|------|----------|-------------|
| 1 | 0.725 | 0.719 |
| 2 | 0.696 | 0.689 |
| 3 | 0.646 | 0.631 |
| 4 | 0.748 | 0.737 |
| 5 | 0.768 | 0.758 |
| **Mean** | **0.717 ± 0.048** | **0.707 ± 0.049** |

**Conclusion:** Zero class near-perfect (0.95 F1), but non-zero classes weak (0.42-0.59 F1). The 3 non-zero bins are too close together to separate reliably.

---

## Phase 4: Hourly (3-Hour Bin) — Two-Stage Classification

**Approach:**  
- Separate the easy problem (zero vs non-zero) from the hard onу
- Кeduce non-zero classes from hight-mid-low  to hight-low bins.

**Architecture:**
- **Stage 1:** GradientBoosting binary — zero vs non-zero (SMOTE balanced)
- **Stage 2:** GradientBoosting binary — low (1-3 trips) vs high (4+) — only on non-zero subset (SMOTE balanced)
- **3 final classes:** zero, low, high

### Stage 1 Results — Zero vs Non-zero (94.1%)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| zero | 0.92 | 0.98 | 0.94 | 985 |
| non-zero | 0.97 | 0.90 | 0.94 | 911 |

Confusion matrix:

|  | Pred: zero | Pred: non-zero |
|--|-----------|---------------|
| **Actual: zero** | 961 | 24 |
| **Actual: non-zero** | 88 | 823 |

### Stage 2 Results — Low vs High (88.1%)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| low (1-3) | 0.90 | 0.96 | 0.93 | 736 |
| high (4+) | 0.77 | 0.55 | 0.64 | 175 |

Confusion matrix:

|  | Pred: high | Pred: low |
|--|-----------|----------|
| **Actual: high** | 96 | 79 |
| **Actual: low** | 29 | 707 |

### Combined 3-Class Results (88.9%)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| zero | 0.92 | 0.98 | 0.94 | 985 |
| low (1-3) | 0.87 | 0.85 | 0.86 | 736 |
| high (4+) | 0.77 | 0.55 | 0.64 | 175 |
| **Overall** | **0.88** | **0.89** | **0.88** | **1896** |

### Cross-Validation (5-fold TimeSeriesSplit)

| Fold | Accuracy | Weighted F1 |
|------|----------|-------------|
| 1 | 0.858 | 0.853 |
| 2 | 0.839 | 0.828 |
| 3 | 0.825 | 0.819 |
| 4 | 0.900 | 0.893 |
| 5 | 0.891 | 0.886 |
| **Mean** | **0.863 ± 0.032** | **0.856 ± 0.033** |

### Top 10 Features

**Stage 1 (zero vs non-zero):**

| Feature | Importance |
|---------|-----------|
| member_ratio | 0.211 |
| lag_56 | 0.109 |
| lag_8 | 0.066 |
| lag_3 | 0.030 |
| year | 0.029 |
| rolling_mean_same_bin_7d | 0.026 |
| rolling_std_8 | 0.025 |
| day_of_month | 0.024|
| dow_sin	 | 0.022 |
| bin_cos	| 0.021 |


## Summary

| Phase | Approach | Classes | Accuracy | Weighted F1 |
|-------|----------|---------|----------|-------------|
| 1 | Daily regression | — | Failed (R² < 0) | — |
| 2 | Daily classification (XGBoost) | 4 | Failed 97.6%  (overfitting)| 0.98 |
| 3 | Hourly single-stage (GradientBoosting) | 4 | 77.0% | 0.76 |
| **4** | **Hourly two-stage (GradientBoosting)** | **3** | **88.9%** | **0.88** |

