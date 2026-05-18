"""
Data loading, quality checks, and cleaning for fraud datasets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path):
    return pd.read_csv(path)


def run_quality_checks(df, target_col):
    duplicate_rows = int(df.duplicated().sum())
    infinite_values = int(np.isinf(df.select_dtypes(include=["number"])).sum().sum())
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    near_constant_cols = [c for c in df.columns if df[c].nunique() < 5]
    class_counts = df[target_col].value_counts()
    total = len(df)
    fraud_count = int(class_counts.get(1, 0))
    fraud_rate = fraud_count / total if total > 0 else 0
    imbalance_ratio = class_counts.get(0, 0) / max(fraud_count, 1)
    return {
        "shape": list(df.shape), "duplicate_rows": duplicate_rows,
        "infinite_values": infinite_values, "constant_columns": constant_cols,
        "near_constant_columns": near_constant_cols,
        "class_0_count": int(class_counts.get(0, 0)),
        "class_1_count": fraud_count, "fraud_rate": fraud_rate,
        "imbalance_ratio": imbalance_ratio,
        "no_skill_baseline": class_counts.max() / total if len(class_counts) > 0 and total > 0 else 0,
    }


def clean_dataset(df):
    df = df.copy()
    before = df.shape[0]
    df = df.drop_duplicates()
    dropped = before - df.shape[0]
    df = df.replace([np.inf, -np.inf], np.nan)
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df = df.drop(columns=const_cols)
    dup_cols = []
    for i in range(df.shape[1]):
        for j in range(i + 1, df.shape[1]):
            if df.iloc[:, i].equals(df.iloc[:, j]):
                dup_cols.append(df.columns[j])
    df = df.drop(columns=dup_cols)
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(exclude=["number"]).columns:
        if len(df[col].dropna()) > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df, {"duplicates_removed": dropped, "constant_columns_removed": const_cols, "duplicate_columns_removed": dup_cols}


def sample_dataset(df, target_col, sample_size, random_state=42):
    original_size = len(df)
    if original_size <= sample_size:
        return df, original_size
    min_class_count = df[target_col].value_counts().min()
    if min_class_count < 2:
        sampled, _ = train_test_split(df, train_size=sample_size, random_state=random_state)
    else:
        sampled, _ = train_test_split(df, train_size=sample_size, stratify=df[target_col], random_state=random_state)
    return sampled, original_size
