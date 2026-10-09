# Mortgage Default Prediction from Account-Balance Behaviour

A customer-level model that estimates the **probability of mortgage default** from anonymised retail-banking data. It covers the full path from raw daily balances to a threshold policy that a credit-risk team could actually use.

---

## The problem

A lender wants to flag mortgage customers who are likely to default, so it can review them and act early. The model has to rank customers well, but it also has to support a **decision**: how many customers to flag, and what that costs.

## Data (not included)

Four anonymised tables, joined at customer level (5,000 customers, **20% default rate**):

| Table | Content |
|---|---|
| `case_eod_balances` | Daily end-of-day balance per product, 2013 – early 2016 (8M+ rows, 7k+ products) |
| `case_cust_prod_mapping` | Customer ↔ product mapping |
| `case_misc_features` | Age group, customer segment, number of products |
| `case_target` | Default flag per customer (defaults observed from 2016 onward) |

The data was supplied privately for this exercise, so it is **not part of this repository**.

---

## What I did

### 1. Exploration — [`01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb)
- Checked data quality: the data is largely complete, with small gaps in `age_group` (~4%) and `cust_segment` (~6%).
- **Demographics show some signal.** Default rates are higher for younger customers. Segment and number of products differentiate only a little.
- **Liquidity stress is the strongest early signal.** About 15% of daily balance records are negative. Defaulters spend **3× more days below zero**, and their average negative-balance ratio is ~37% against ~10% for non-defaulters.

### 2. Feature engineering — [`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb), [`src/feature_engineering.py`](src/feature_engineering.py)
- **Temporal cut-off at 2015-12-31.** The target is observed from 2016, so all features use only data before that date. This prevents look-ahead bias.
- Two-step aggregation: balance statistics are computed **per product over time** and then **aggregated per customer** across products.
  - Level & volatility: `bal_mean_mean`, `bal_std_mean`, `bal_min_mean`, `bal_max_mean`
  - Stress indicators: `neg_days_sum`, `neg_ratio_mean`
  - Coverage: `n_days_sum`

### 3. Modelling — [`03_modeling.ipynb`](notebooks/03_modeling.ipynb)
- scikit-learn `Pipeline` + `ColumnTransformer`: median imputation and scaling for numeric features, mode imputation and one-hot encoding for categoricals.
- Stratified 70/30 split. Hyperparameters tuned with `RandomizedSearchCV` (3-fold, ROC-AUC).

| Model | Test ROC-AUC |
|---|---|
| Logistic Regression (baseline) | 0.805 |
| XGBoost (tuned) | 0.829 |
| **Random Forest (tuned) — selected** | **0.834** |

- Average Precision **0.579** (against a 0.20 base rate). Cross-validation and test scores are consistent (CV 0.831 → test 0.834), so overfitting is limited.
- **Top drivers:** mean balance, minimum balance, negative-balance ratio and negative-balance days. Behaviour matters much more than demographics.

### 4. From scores to decisions — [`04_evaluation_and_business_insights.ipynb`](notebooks/04_evaluation_and_business_insights.ipynb)

Threshold analysis on the test set:

| Threshold | Precision | Recall | Customers flagged |
|---|---|---|---|
| 0.2 | 0.47 | 0.78 | 34% |
| 0.3 | 0.51 | 0.66 | 26% |
| **0.4** | **0.56** | **0.56** | **20%** |
| 0.5 | 0.59 | 0.43 | 15% |

- **0.4 is a balanced operating point**: precision ≈ recall, and the flagged share roughly matches the portfolio default rate.
- **Cost simulation.** With a missed default (FN) costing €10,000 and an unnecessary review (FP) costing €500, total cost is lowest at the **lowest threshold**. When the losses are this asymmetric, the threshold should come from the cost structure and risk appetite, not from F1.

---

## Key takeaways
- **Behavioural balance features beat static demographics** for predicting default.
- **Negative-balance frequency and depth** work as early-warning indicators.
- **Threshold choice is a business decision.** The same model supports a conservative policy or a broad screening policy depending on review capacity and loss asymmetry.

## Limitations & next steps
- Random (stratified) split → add **out-of-time validation**.
- Balance trajectories are summarised as static statistics → model **trends and recent behaviour** (e.g. last 3 / 6 months, slopes).
- Add **cost-sensitive training** and **probability calibration** before using the scores for pricing or provisioning.

---

## Repository structure

```
mortgage-default-prediction/
├── data/
│   ├── case_data/      # raw CSVs go here (not versioned)
│   └── processed/      # df_model_v1.csv is written here
├── model/              # trained model + metadata (generated, not versioned)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation_and_business_insights.ipynb
├── src/
│   ├── data_io.py
│   ├── eda_utils.py
│   └── feature_engineering.py
├── environment.yml
└── README.md
```

## How to run

```bash
conda env create -f environment.yml
conda activate mortgage-default-ml
jupyter lab
```

Put `case_eod_balances.csv`, `case_cust_prod_mapping.csv`, `case_misc_features.csv` and `case_target.csv` in `data/case_data/`, then run the notebooks in order. In `02_feature_engineering.ipynb`, set `to_save = True` to write the modelling table.

**Tech:** Python · pandas · scikit-learn · XGBoost · matplotlib / seaborn · Jupyter
