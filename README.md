# fraud-analysis
Financial fraud is a persistent threat to modern payment systems. This project builds and compares machine learning models across three distinct fraud datasets to identify which algorithms generalize best and how dataset characteristics affect model performance.

End-to-end fraud detection analysis across three financial datasets using multiple classifiers, SMOTE-based imbalance handling, and cost-based optimization.


## Datasets

This notebook uses three fraud-related datasets supplied for the project:

1. `creditcard.csv`: https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection
2. `onlinefraud.csv`: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
3. `Base.csv`: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/


| Dataset | Source | Size | Fraud Rate | Imbalance |
|---------|--------|------|------------|-----------|
| **Credit Card Fraud** | Kaggle (PCA-transformed) | 284,807 | 0.17% | 578:1 |
| **Online Payment Fraud** | Kaggle | 200,000 (sampled) | 0.63% | 158:1 |
| **Bank Account Fraud** | Base.csv | 1,000,000 | 1.14% | 87:1 |

## How to Run

### Full Pipeline (3 datasets, 5 models)

```bash
python run_pipeline.py
```

Runs EDA -> preprocessing -> SMOTE -> train 5 models -> confusion matrices -> feature importance -> cross-dataset summary for all three datasets. Outputs to `pipeline_plots/` and `pipeline_results.csv`.

### Credit Card Deep Analysis (sections 1.6–1.11)

```bash
python credit_card_deep_analysis.py
```

Deep-dive into the Credit Card dataset: multivariate correlations, weirdness scores, SMOTE/ADASYN/undersample experiments, baselines, and class-weight comparisons.

### Credit Card Deep Analysis Part 2 (sections 1.12–1.17)

```bash
python credit_card_deep_analysis_part2.py
```

Model tuning (GridSearchCV across RF/XGB/LR), threshold sweep (0.05–0.95), cost optimization with Z = cost_fn × FN + cost_fp × FP across 7 cost ratios (1:1 to 1000:1), and threshold stability refinement.

## Pipeline Overview

`run_pipeline.py` processes each dataset through a consistent pipeline:

```
Load CSV -> Data Quality Checks -> Class Distribution -> 
Correlation Analysis -> Skew Detection -> Feature Distributions by Class ->
Categorical Fraud Rates -> Log Transform (skew > 1) -> 
Stratified Split (80/20) -> SMOTE Oversampling -> StandardScaler ->
Train 5 Models -> Evaluate -> Confusion Matrices -> Feature Importance ->
Cross-Dataset Summary
```

### Models Compared

| Model | Config |
|-------|--------|
| Dummy (most_frequent) | Baseline - always predicts majority class |
| Dummy (stratified) | Baseline - samples from training distribution |
| Logistic Regression | `class_weight="balanced"`, max_iter=1000 |
| Random Forest | 200 trees, `class_weight="balanced"` |
| XGBoost | 200 trees, max_depth=6, subsample=0.8 |

## Key Findings

### 1. Accuracy is Misleading on Imbalanced Data

Dummy classifiers achieve 99%+ accuracy on all three datasets by simply predicting "non-fraud" for every transaction. Precision, recall, and F1-score are the only meaningful metrics.

### 2. Tree Models Dominate on PCA Features

| Dataset | Best Model | F1 | Recall | Precision |
|---------|-----------|-----|--------|-----------|
| Credit Card | Random Forest | 0.837 | 0.786 | 0.895 |
| Online Payment | XGBoost | 0.990 | 0.983 | 0.997 |
| Bank Account | Random Forest | 0.774 | 0.782 | 0.765 |

Random Forest and XGBoost consistently outperform Logistic Regression, which struggles with the non-linear separability of PCA-transformed features.

### 3. SMOTE Significantly Improves Recall

Without SMOTE, Logistic Regression catches near-zero fraud cases. SMOTE oversampling increases fraud recall from ~0% to >70% across all models. Experiments with ADASYN and RandomUnderSampling show SMOTE provides the best precision-recall balance.

### 4. Threshold Tuning is Critical

The default 0.5 decision threshold is rarely optimal for imbalanced data. Sweeping thresholds from 0.05–0.95 reveals:

- Lower thresholds (0.10–0.25) improve recall at the cost of precision
- The "optimal" threshold depends entirely on business cost preferences
- A tuned threshold beats default 0.5 on every metric for every model

### 5. Cost-Based Optimization (Z)

Using a cost function **Z = $100 × FN + $1 × FP** to reflect business impact:

| Model | Threshold | Fraud Caught | False Alarms | Total Cost |
|-------|-----------|-------------|--------------|------------|
| RF    | 0.20      | 89/98 (91%) | 128          | $1,028     |
| XGB   | 0.15      | 87/98 (89%) | 66           | $1,166     |
| LR    | 0.94      | 87/98 (89%) | 83           | $1,183     |

**Optimal: Random Forest at threshold 0.20** - catches 91% of fraud with 128 false alarms per 57,000 transactions. The optimal threshold is stable (±0.02) with <1% cost variation.

Cost ratios from 1:1 to 1000:1 show a consistent pattern: as fraud cost increases, optimal threshold drops (favoring more catches at the expense of more false alarms).

### 6. Feature Separation

PCA features V14, V10, V12, V4, and V11 show the strongest separation between fraud and non-fraud (KS test statistic > 0.75). The "weirdness score" - average distance from the non-fraud centroid - is 4.6× higher for fraud cases, suggesting it could serve as a useful meta-feature.

### 7. Stability

The cost-optimal threshold is robust: cost varies by less than 1% for thresholds within ±0.02 of the optimum, and the optimal point is identical between coarse (0.01 step) and fine (0.005 step) searches.

## Plots

The `pipeline_plots/` directory contains 65+ plots:

- **Per-dataset**: Class distribution, correlations, skew, feature-by-class distributions
- **Per-model**: Confusion matrices, feature importance
- **Summary**: Cross-dataset model comparison (accuracy, precision, recall, F1, ROC-AUC)
- **Deep analysis**: Multivariate correlations, threshold tuning, cost optimization, stability refinement

## Results

All numeric results are saved to `pipeline_results.csv` with columns: dataset, model, accuracy, precision, recall, f1, roc_auc.

Analysis findings are saved as structured text files in `analysis_findings/` with one file per section (1.6 through 1.17).
