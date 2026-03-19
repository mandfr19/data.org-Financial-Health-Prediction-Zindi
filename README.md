# Financial Health Index (FHI) Prediction
## Zindi Competition — Southern African SME Financial Health Classification

---

## Results

| Split | Score |
|-------|-------|
| Public Leaderboard | 0.8847 |
| **Private Leaderboard** | **0.8860** ✅ |

---

## Overview

This repository contains the solution for the **Financial Health Index Prediction Challenge**, a Zindi competition focused on predicting the financial well-being of small and medium-sized enterprises (SMEs) across four Southern African countries: Eswatini, Lesotho, Zimbabwe, and Malawi.

The task is a **multiclass classification problem** — predicting whether a business has **Low**, **Medium**, or **High** financial health — evaluated using **Macro F1 Score**.

---

## Problem Statement

Traditional measures like revenue or profit do not fully capture an SME's financial well-being. This competition introduces a holistic **Financial Health Index (FHI)** — a composite measure reflecting resilience, savings habits, and access to finance across four dimensions:

- Savings and assets
- Debt and repayment ability
- Resilience to shocks
- Access to credit and financial services

Participants build machine learning models to predict FHI using socio-economic and business survey data.

---

## Repository Structure

```
├── winning_solution.ipynb    # ⭐ Main solution — best private LB (0.8860)
├── experiments.ipynb         # Experimental notebook — additional approaches explored
├── README.md
├── VariableDefinitions.csv
├── requirements.txt
├── .gitignore
└── outputs/
    ├── submission_final.csv
    ├── experiment_log.json
    └── README.md
```

### Solution Files

| File | Description | Private LB |
|------|-------------|------------|
| [`winning_solution.ipynb`](./winning_solution.ipynb) | **Main solution** — clean preprocessing, feature engineering, 3-model ensemble (XGB + LGB + CAT), Optuna tuning, SMOTE, threshold optimization | **0.8860** ✅ |
| [`experiments.ipynb`](./experiments.ipynb) | Extended experiments — pseudo-labeling, target encoding, MLP ensemble, additional feature engineering | 0.8814 |

> The winning private LB solution is the simpler 3-model ensemble in `winning_solution.ipynb`. The experiments notebook achieved a higher public LB score (0.8969) through additional techniques, but the simpler pipeline generalized better to the private test set — a key lesson about overfitting to the public leaderboard.

---

## Dataset

| Split | Rows | Features |
|-------|------|----------|
| Train | 9,618 | 38 |
| Test  | 2,405 | 37 |

**Feature types:**
- Categorical survey responses (Yes/No, Have now/Never had, etc.)
- Numeric (owner age, income, expenses, turnover, business age)
- Country (Eswatini, Lesotho, Malawi, Zimbabwe)

**Target distribution (train):**

| Class | Count | Percentage |
|-------|-------|------------|
| Low | 6,280 | 65.3% |
| Medium | 2,868 | 29.8% |
| High | 470 | 4.9% |

The severe class imbalance — particularly the minority High class — was the primary challenge throughout the competition.

---

## Data Quality Issues Found

A thorough data audit revealed several encoding issues that were silently corrupting features before any modeling. Fixing these was the single most impactful change in the entire competition.

| Issue | Description | Fix Applied |
|-------|-------------|-------------|
| Apostrophe variants | Curly `'` (Unicode 8217) vs straight `'` (Unicode 39) caused silent NaN mapping across 9 status columns | Normalized to straight apostrophe |
| Mixed values | `current_problem_cash_flow` contained `"0"` alongside `"Yes"`/`"No"` | Mapped `"0"` → `"No"` |
| Don't know variants | Multiple inconsistent representations (`"Don't know"`, `"Don't Know"`, `"Don?t know"`, etc.) | Unified to single `-1` encoding |
| Invisible unicode | `perception_insurance_important` contained zero-width character `\u200e` | Stripped via whitespace normalization |
| Age placeholders | `owner_age` values of 99 and 103 are survey refusal codes, not real ages | Mapped to NaN then imputed |
| Inconsistent records | `keeps_financial_records` had `"Yes, always"` and `"Yes, sometimes"` | Unified to `"Yes"` |

---

## Winning Solution Pipeline (`winning_solution.ipynb`)

```
Raw Survey Data
      │
      ▼
Data Cleaning & Preprocessing
      │  - Apostrophe normalization (curly → straight)
      │  - Standardize all "don't know" variants → -1
      │  - Status columns → ordinal integers
      │  - Binary Yes/No → 1/0/-1
      │  - Country → one-hot encoding
      │
      ▼
Feature Engineering
      │  - Log-transformed financials (handles extreme currency skew)
      │  - Financial ratios (profit proxy, expense ratio, income efficiency)
      │  - Business age in total months
      │  - Missing value flags (15 high-missingness columns)
      │  - Count of active financial products
      │  - Positive attitude score
      │
      ▼
SMOTE Oversampling (per fold, training data only)
      │  - strategy="minority" — oversample High to match Medium count
      │  - Validation fold always uses original, unaugmented data
      │
      ▼
5-Fold Stratified CV — Optuna Tuned OOF Training (50 trials each)
      │  - XGBoost
      │  - LightGBM
      │  - CatBoost
      │
      ▼
Weighted Ensemble
      │  - Weights proportional to each model's OOF F1 score
      │
      ▼
Threshold Optimization (Nelder-Mead via scipy.optimize)
      │  - Per-class probability multipliers optimized directly for Macro F1
      │
      ▼
Submission
```

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `log_personal_income` | Log1p transform — handles extreme income skew |
| `log_business_expenses` | Log1p transform of expenses |
| `log_business_turnover` | Log1p transform of turnover |
| `profit_proxy` | Turnover minus expenses |
| `log_profit_proxy` | Log1p of profit proxy |
| `expense_ratio` | Expenses / (turnover + ε) |
| `income_to_turnover` | Personal income / (turnover + ε) |
| `income_to_expenses` | Personal income / (expenses + ε) |
| `total_business_months` | Business age years × 12 + months |
| `missing_*` (×15) | Binary flags for high-missingness columns |
| `n_financial_products` | Count of currently active financial products |
| `attitude_score` | Sum of positive owner attitudes (0–3) |

**Key insight — Missing Value Flags:** Many columns have 20–47% missingness due to country-level survey differences. The pattern of who didn't answer is itself highly informative — `missing_business_age_years` was consistently the second most important SHAP feature, reflecting that Lesotho respondents have systematic missingness which correlates strongly with Low FHI.

---

## Models

All models were trained with 5-fold stratified CV, SMOTE per fold, and Optuna hyperparameter tuning (50 trials each).

### XGBoost
- Histogram-based algorithm (`tree_method="hist"`)
- Balanced sample weights computed after SMOTE (not before — class distribution changes after resampling)
- Search space: n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda

### LightGBM
- Leaf-wise tree growth with `class_weight="balanced"`
- Early stopping via LGB callbacks (API differs from XGBoost)
- Search space: n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda

### CatBoost
- `auto_class_weights="Balanced"` for imbalance handling
- `eval_metric="TotalF1"` — optimizes directly for the competition metric during training
- Search space: iterations, learning_rate, depth, l2_leaf_reg, bagging_temperature, random_strength

---

## Ensemble & Threshold Optimization

**Weighted average** where each model's weight is proportional to its OOF F1 score, followed by per-class threshold optimization using Nelder-Mead:

$$\hat{y} = \arg\max_k \frac{p_k}{t_k}$$

where $p_k$ is the predicted probability for class $k$ and $t_k$ is the learned threshold. This approach added +0.0086 macro F1 over plain argmax at zero retraining cost.

---

## Experiment Tracking

Key milestones across the competition:

| Approach | OOF F1 | Public LB | Private LB |
|----------|--------|-----------|------------|
| Baseline XGBoost (clean data) | 0.7915 | 0.8703 | — |
| Baseline + SMOTE + thresholds | 0.7989 | 0.8764 | — |
| Clean XGB + SMOTE + Optuna | 0.8033 | 0.8839 | — |
| **3-model ensemble + SMOTE + Optuna + thresholds** | **0.8082** | **0.8847** | **0.8860** ✅ |
| 4-model + pseudo-labeling + target encoding | 0.8049 | 0.8919 | — |
| Final + fintech/insurance features | 0.8069 | 0.8969 | — |

The private LB winner was the simpler 3-model ensemble. The experiments that improved the public LB score — pseudo-labeling, target encoding, and additional feature groups — did not generalize to the private test set.

---

## What Worked

| Strategy | Impact |
|----------|--------|
| Full data cleaning (apostrophe fix + encoding fixes) | ✅ Most impactful single change |
| Log-transformed financials | ✅ Positive |
| Missing value flags | ✅ Positive — missingness pattern is informative |
| SMOTE minority oversampling | ✅ Positive |
| Optuna hyperparameter tuning | ✅ Positive |
| Weighted ensemble (3 models) | ✅ Positive |
| Threshold optimization | ✅ Positive — free +0.0086 F1 |

## What Did Not Generalize to Private LB

| Strategy | Public LB Effect | Lesson |
|----------|-----------------|--------|
| Pseudo-labeling | +0.0072 public | Overfit to public test set |
| Target encoding for country | +0.0072 public | One-hot encoding was sufficient |
| Fintech/insurance feature group | +0.0122 public | Added noise on private data |
| MLP as 4th ensemble member | Small positive | Tree models were sufficient |

The core lesson: on survey-based tabular data with high missingness, clean preprocessing and well-tuned tree models generalize better than complex feature engineering and semi-supervised techniques.

---

## Reproducibility Note

XGBoost results are hardware-dependent:

| Hardware | Effect |
|----------|--------|
| GPU (`device="cuda"`) | Best results — used for the winning submission |
| CPU (`device="cpu"`) | ~0.003 lower |

The difference arises from floating-point parallelization in XGBoost's `hist` method — GPU and CPU use different numerical precision which produces different split points. LightGBM and CatBoost are unaffected by hardware. `winning_solution.ipynb` was run on a Kaggle T4 GPU.

---

## Requirements

```
python>=3.9
numpy
pandas
scikit-learn
imbalanced-learn
lightgbm
xgboost
catboost
optuna
scipy
```

Install with:
```bash
pip install numpy pandas scikit-learn imbalanced-learn lightgbm xgboost catboost optuna scipy
```

---

## Future Improvements

Based on post-competition analysis and insights from other participants:

- **NFKC Unicode normalization** — more comprehensive than our targeted apostrophe fix, catches all hidden character variants in one pass using `unicodedata.normalize('NFKC', x)`
- **Multi-seed ensembling** — averaging predictions across multiple random seeds (e.g. 1, 78, 93, 225) reduces variance from initialization, typically worth +0.003 to +0.008
- **Ordinal ensembling** — mapping predictions to [0, 1, 2] and averaging ordinally before rounding respects the natural Low < Medium < High ordering
- **More Optuna trials (200+)** — 50 trials per model leaves room for better hyperparameter search

---

## Competition Details

- **Platform:** [Zindi](https://zindi.africa/competitions/dataorg-financial-health-prediction-challenge)
- **Task:** Multiclass classification (Low / Medium / High)
- **Metric:** Macro F1 Score
- **Data:** Survey data from Eswatini, Lesotho, Zimbabwe, Malawi
- **Best public LB (competition):** 0.9211
- **Our best public LB:** 0.8969
- **Our best private LB:** 0.8860
