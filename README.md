# Financial Health Index (FHI) Prediction
## Zindi Competition — Southern African SME Financial Health Classification

---

## Overview

This repository contains the solution for the **Financial Health Index Prediction Challenge**, a Zindi competition focused on predicting the financial well-being of small and medium-sized enterprises (SMEs) across four Southern African countries: Eswatini, Lesotho, Zimbabwe, and Malawi.

The task is a **multiclass classification problem** — predicting whether a business has **Low**, **Medium**, or **High** financial health — evaluated using **Macro F1 Score**.

**Best Leaderboard Score: 0.8969**

---

## Problem Statement

Traditional measures like revenue or profit do not fully capture an SME's financial well-being. This competition introduces a holistic **Financial Health Index (FHI)** — a composite measure reflecting resilience, savings habits, and access to finance across four dimensions:

- Savings and assets
- Debt and repayment ability
- Resilience to shocks
- Access to credit and financial services

Participants build machine learning models to predict FHI using socio-economic and business survey data.

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
- Low: 6,280 (65.3%) — dominant class
- Medium: 2,868 (29.8%)
- High: 470 (4.9%) — minority class, hardest to predict

---

## Data Quality Issues Found

A thorough data audit revealed several encoding issues requiring targeted fixes:

| Issue | Description | Fix Applied |
|-------|-------------|-------------|
| Apostrophe variants | Curly `'` (Unicode 8217) vs straight `'` (Unicode 39) caused silent NaN mapping across 9 status columns | Normalized to straight apostrophe |
| Mixed values | `current_problem_cash_flow` contained `"0"` alongside `"Yes"`/`"No"` | Mapped `"0"` → `"No"` |
| Don't know variants | Multiple inconsistent representations across columns | Unified to single `-1` encoding |
| Invisible unicode | `perception_insurance_important` contained zero-width character `\u200e` | Stripped via whitespace normalization |
| Case inconsistency | `"Don't Know"` vs `"Don't know"` (capital K) | Unified during normalization |
| Near-constant features | `uses_informal_lender` had near-zero variance due to encoding errors | Fixed by apostrophe normalization |

---

## Solution Pipeline

```
Raw Data
    │
    ▼
Data Cleaning & Preprocessing
    │  - Apostrophe normalization
    │  - Status columns → ordinal (_ord suffix)
    │  - Binary Yes/No → 1/0/-1
    │  - Country → one-hot
    │
    ▼
Feature Engineering
    │  - Log-transformed financials
    │  - Financial ratios (profit proxy, expense ratio)
    │  - Business age in total months
    │  - Missing value flags
    │  - Financial product counts
    │  - Fintech adoption score
    │  - Insurance sophistication score
    │  - Formal vs informal finance ratio
    │  - Business vs personal income ratio
    │
    ▼
Target Encoding for Country
    │  - Replaces one-hot with P(High | country)
    │  - Out-of-fold encoding prevents leakage
    │
    ▼
SMOTE Oversampling (per fold)
    │  - strategy="minority" — oversample High to match Medium
    │  - Applied on training fold only
    │
    ▼
10-Fold Stratified CV — OOF Training
    │  - XGBoost
    │  - LightGBM
    │  - CatBoost
    │  - MLP (scaled features)
    │
    ▼
Pseudo-Labeling
    │  - Threshold = 0.95 confidence
    │  - High-confidence test predictions added to training
    │  - All models retrained on augmented dataset
    │
    ▼
Weighted Ensemble
    │  - Weights proportional to OOF F1 score per model
    │
    ▼
Threshold Optimization
    │  - Per-class thresholds via Nelder-Mead (scipy.optimize)
    │  - Optimizes directly for Macro F1
    │
    ▼
Final Submission
```

---

## Feature Engineering

### Original Features (after preprocessing)
- 10 status columns → ordinal encoded with `_ord` suffix
- 19 binary columns → 1/0/-1
- 4 country one-hot columns (later target encoded)
- 6 numeric columns

### Engineered Features

| Feature | Description |
|---------|-------------|
| `log_personal_income` | Log1p transform of personal income |
| `log_business_expenses` | Log1p transform of business expenses |
| `log_business_turnover` | Log1p transform of business turnover |
| `profit_proxy` | Turnover minus expenses |
| `log_profit_proxy` | Log1p of profit proxy |
| `expense_ratio` | Expenses / (turnover + ε) |
| `income_to_turnover` | Personal income / (turnover + ε) |
| `income_to_expenses` | Personal income / (expenses + ε) |
| `total_business_months` | Business age years × 12 + months |
| `missing_*` | Binary flags for 15 high-missingness columns |
| `n_financial_products` | Count of currently active financial products |
| `attitude_score` | Sum of positive owner attitudes |
| `business_vs_personal_ratio` | Turnover / (personal income + 1) |
| `log_business_vs_personal_ratio` | Log1p of above |
| `fintech_adoption_score` | Count of active digital financial tools |
| `insurance_sophistication_score` | Count of active insurance products |
| `formal_credit_score` | Count of active formal credit products |
| `informal_finance_reliance` | Count of active informal finance products |
| `formal_vs_informal_ratio` | Formal credit / (informal + 1) |
| `overall_financial_access` | Weighted composite of fintech + insurance + formal credit |

### Target Encoding for Country
Country is highly predictive of FHI — Eswatini has 11.5% High businesses vs Lesotho's 0.3%. One-hot country columns are replaced with `P(High | country)` computed out-of-fold to prevent leakage.

---

## Models

All models trained with:
- 10-fold stratified CV
- SMOTE minority oversampling per fold
- Early stopping on validation loss

### XGBoost
```python
n_estimators=800, learning_rate=0.03, max_depth=6,
min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
gamma=0.01, reg_alpha=0.1, reg_lambda=1.0,
tree_method="hist", device="cuda"
```

### LightGBM
```python
n_estimators=800, learning_rate=0.03, num_leaves=80,
max_depth=8, min_child_samples=20, subsample=0.8,
colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
class_weight="balanced"
```

### CatBoost
```python
iterations=800, learning_rate=0.03, depth=6,
l2_leaf_reg=1.0, bagging_temperature=0.5,
border_count=128, auto_class_weights="Balanced"
```

### MLP
```python
hidden_layer_sizes=(256, 128, 64), activation="relu",
solver="adam", alpha=0.001, max_iter=500,
early_stopping=True, validation_fraction=0.1
# Note: requires StandardScaler preprocessing
# class_weight not supported — imbalance handled via SMOTE
```

---

## Pseudo-Labeling

High-confidence test predictions (max probability ≥ 0.95) are added to the training set before final retraining. This leverages unlabeled test data to improve generalization.

**Key findings from experimentation:**
- Threshold 0.95 was optimal — lower thresholds (0.93) added noisy labels, higher thresholds (0.97, 0.98) added too few samples
- Iterative pseudo-labeling (2 rounds) hurt performance due to compounding label errors — single round only

---

## Ensemble Strategy

Weighted average ensemble where each model's weight is proportional to its OOF F1 score. Per-class probability thresholds optimized via Nelder-Mead to maximize Macro F1.

**Strategies tested and rejected:**
- Simple average — marginal difference
- Rank averaging — hurt performance
- Power weighted average — negligible difference
- Calibrated weighted average — hurt performance
- Tree models only (no MLP) — hurt performance

---

## Experiment Tracking

All experiments tracked with OOF F1 and Leaderboard F1:

| Approach | OOF F1 | LB F1 |
|----------|--------|-------|
| Baseline XGBoost (clean data) | 0.7915 | 0.8703 |
| Baseline + SMOTE + thresholds | 0.7989 | 0.8764 |
| Baseline + SMOTE + Optuna | 0.8033 | 0.8839 |
| 3-model ensemble + SMOTE | 0.8073 | 0.8821 |
| 4-model + pseudo + target enc | 0.8049 | 0.8919 |
| **4-model + pseudo + target enc + new features** | **0.8069** | **0.8969** ✅ |

---

## What Worked

| Strategy | Impact |
|----------|--------|
| Apostrophe normalization | ✅ Unlocked true signal in 9 status columns |
| Feature engineering (log transforms, ratios) | ✅ Positive |
| SMOTE minority oversampling | ✅ Positive |
| Target encoding for country | ✅ Positive |
| Pseudo-labeling at 0.95 threshold | ✅ Biggest single jump |
| MLP as 4th ensemble member | ✅ Small positive diversity gain |
| 10-fold CV | ✅ More stable OOF estimates |
| Financial access features (fintech, insurance, formal/informal) | ✅ Positive |

## What Did Not Work

| Strategy | Impact |
|----------|--------|
| Country-normalized financial features | ❌ Added noise |
| Interaction features (age × turnover etc.) | ❌ Added noise |
| Meta-learner stacking | ❌ Underfit |
| Iterative pseudo-labeling (round 2) | ❌ Compounded errors |
| Tighter pseudo threshold (0.97, 0.98) | ❌ Too few samples |
| Probability calibration | ❌ Models already well-calibrated |
| Rank averaging ensemble | ❌ Lost probability information |

---

## Reproducibility Note

Results are hardware-dependent for XGBoost:

| Hardware | LB F1 |
|----------|-------|
| GPU (`device="cuda"`) | **0.8969** |
| CPU (`device="cpu"`) | ~0.8940 |

LightGBM, CatBoost, and MLP are unaffected by hardware. The difference arises from floating point parallelization in XGBoost's `hist` tree method — GPU and CPU use different numerical precision which produces different split points.

To reproduce the best result, run on a machine with a GPU and set `device="cuda"` in XGBoost params.

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
shap
matplotlib
```

Install with:
```bash
pip install numpy pandas scikit-learn imbalanced-learn lightgbm xgboost catboost optuna scipy shap matplotlib
```

---

## Repository Structure

```
├── notebook.ipynb          # Final solution notebook (13 cells)
├── outputs/
│   ├── oof_xgb.npy         # XGBoost OOF probabilities
│   ├── oof_lgb.npy         # LightGBM OOF probabilities
│   ├── oof_cat.npy         # CatBoost OOF probabilities
│   ├── oof_mlp.npy         # MLP OOF probabilities
│   ├── oof_xgb_pl.npy      # XGBoost OOF (pseudo-labeled)
│   ├── oof_lgb_pl.npy      # LightGBM OOF (pseudo-labeled)
│   ├── oof_cat_pl.npy      # CatBoost OOF (pseudo-labeled)
│   ├── oof_mlp_pl.npy      # MLP OOF (pseudo-labeled)
│   ├── submission_final.csv
│   └── experiment_log.json
├── Train.csv
├── Test.csv
├── SampleSubmission.csv
├── VariableDefinitions.csv
└── README.md
```

---

## Future Improvements

Based on techniques shared by top competitors:

- **NFKC Unicode normalization** — more comprehensive than targeted apostrophe fix, catches all hidden character variants in one pass
- **Multi-seed ensembling** — averaging predictions across seeds (e.g. 1, 78, 93, 225) reduces variance from random initialization
- **Ordinal ensembling** — mapping predictions to [0, 1, 2] and averaging ordinally before rounding respects the natural Low < Medium < High ordering
- **Optuna tuning with 200+ trials** — we used fixed params; tuned params could add +0.005 to +0.010

---

## Competition Details

- **Platform:** Zindi
- **Task:** Multiclass classification (Low / Medium / High)
- **Metric:** Macro F1 Score
- **Countries:** Eswatini, Lesotho, Zimbabwe, Malawi
- **Best public LB score:** 0.9211
- **Our best LB score:** 0.8969
