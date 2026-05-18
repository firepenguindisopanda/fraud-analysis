"""
Exploratory Data Analysis for fraud datasets.
Produces statistics output to JSON for report consumption.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def compute_correlations(df, target_col, top_n=10):
    numeric = df.select_dtypes(include=["number"]).columns
    if target_col not in numeric or len(numeric) < 2:
        return {}
    corr = df[numeric].corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    return {feat: round(float(val), 4) for feat, val in corr.head(top_n).items()}


def compute_skewness(df, target_col, top_n=8):
    numeric = df.select_dtypes(include=["number"]).columns
    skip = [target_col] if target_col in numeric else []
    skewed = df[numeric].skew().drop(skip, errors="ignore").abs().sort_values(ascending=False)
    return {feat: round(float(val), 4) for feat, val in skewed.head(top_n).items()}


def compute_feature_distributions(df, target_col, features):
    result = {}
    for feat in features:
        if feat not in df.columns or not pd.api.types.is_numeric_dtype(df[feat]):
            continue
        subsets = {c: df[df[target_col] == c][feat].dropna() for c in sorted(df[target_col].unique())}
        stats = {}
        for c, s in subsets.items():
            if len(s) == 0:
                continue
            stats[f"class_{c}"] = {"count": int(len(s)), "mean": round(float(s.mean()), 4), "median": round(float(s.median()), 4), "std": round(float(s.std()), 4)}
        if 0 in subsets and 1 in subsets and len(subsets[0]) > 0 and len(subsets[1]) > 0:
            ks_stat, ks_p = ks_2samp(subsets[0], subsets[1])
            stats["ks_test"] = {"statistic": round(float(ks_stat), 4), "p_value": float(ks_p)}
        result[feat] = stats
    return result


def compute_categorical_fraud_rates(df, target_col, top_n=10):
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    result = {"overall_rate": round(float(df[target_col].mean()), 4)}
    for col in cat_cols[:3]:
        rates = df.groupby(col)[target_col].mean().sort_values(ascending=False).head(top_n)
        counts = df.groupby(col)[target_col].count()
        result[col] = {str(cat): {"rate": round(float(rate), 4), "count": int(counts[cat])} for cat, rate in rates.items()}
    return result


def run_eda(df, target_col, numeric_features=None):
    results = {"correlations": compute_correlations(df, target_col), "skewness": compute_skewness(df, target_col)}
    if numeric_features:
        results["feature_distributions"] = compute_feature_distributions(df, target_col, numeric_features)
    results["categorical_fraud_rates"] = compute_categorical_fraud_rates(df, target_col)
    return results
