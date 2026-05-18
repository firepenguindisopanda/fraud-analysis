"""
Preprocessing: log transforms, encoding, splitting, SMOTE, scaling.
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def log_transform_skewed(df, threshold=1.0):
    df = df.copy()
    transformed = []
    for col in df.select_dtypes(include=["number"]).columns:
        if abs(df[col].skew()) > threshold:
            df[col] = np.log1p(df[col])
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            transformed.append(col)
    return df, transformed


def encode_categoricals(df):
    return pd.get_dummies(df, drop_first=True)


def impute_missing(df):
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df


def prepare_data(df, target_col, test_size=0.2, random_state=42, apply_smote=True):
    df = df.copy()
    X = df.drop(columns=[target_col])
    y = df[target_col]
    info = {}
    X, log_transformed = log_transform_skewed(X)
    info["log_transformed"] = log_transformed
    X = encode_categoricals(X)
    X = impute_missing(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    info["train_size"] = len(X_train)
    info["test_size"] = len(X_test)
    info["train_fraud_rate"] = float(y_train.mean())
    info["test_fraud_rate"] = float(y_test.mean())
    if apply_smote and y_train.sum() > 0:
        smote = SMOTE(random_state=random_state)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        info["after_smote"] = dict(Counter(y_train_smote))
    else:
        X_train_smote, y_train_smote = X_train.copy(), y_train.copy()
        info["after_smote"] = dict(Counter(y_train_smote))
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_smote), columns=X_train_smote.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_smote, X_test, X_train_scaled, X_test_scaled, y_train_smote, y_test, info
