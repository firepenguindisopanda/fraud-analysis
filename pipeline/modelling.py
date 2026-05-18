"""
Model training, evaluation, and metrics computation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier


def get_models():
    return {
        "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent", random_state=42),
        "Dummy (stratified)": DummyClassifier(strategy="stratified", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42, verbosity=0),
    }


def get_training_data(model_name, X_train, X_test, X_train_scaled, X_test_scaled):
    if "Dummy" in model_name:
        return X_train, X_test
    elif model_name == "Logistic Regression":
        return X_train_scaled, X_test_scaled
    else:
        return X_train, X_test


def train_and_evaluate(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    models = get_models()
    rows = []
    fitted_models = {}
    for model_name, model in models.items():
        train_X, test_X = get_training_data(model_name, X_train, X_test, X_train_scaled, X_test_scaled)
        model.fit(train_X, y_train)
        y_pred = model.predict(test_X)
        y_prob = model.predict_proba(test_X)[:, 1] if hasattr(model, "predict_proba") else None
        rows.append({"dataset": None, "model": model_name, "accuracy": accuracy_score(y_test, y_pred), "precision": precision_score(y_test, y_pred, zero_division=0), "recall": recall_score(y_test, y_pred, zero_division=0), "f1": f1_score(y_test, y_pred, zero_division=0), "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan})
        fitted_models[model_name] = {"model": model, "y_pred": y_pred, "y_prob": y_prob}
    return pd.DataFrame(rows), fitted_models


def compute_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1), "precision": tp / max(tp + fp, 1), "fpr": fp / max(fp + tn, 1)}


def compute_roc_pr_curves(y_test, y_prob):
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_prob)
    return {"roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist()}, "pr": {"precision": precision_vals.tolist(), "recall": recall_vals.tolist(), "thresholds": pr_thresholds.tolist()}}


def get_feature_importance(model, feature_names, top_n=15):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_type = "gini"
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1 and coef.shape[0] == 1:
            coef = coef[0]
        importances = np.abs(coef)
        importance_type = "coefficient"
    else:
        return [], ""
    if len(importances) != len(feature_names):
        return [], ""
    indices = np.argsort(importances)[-top_n:]
    top_importances = importances[indices][::-1]
    top_names = [str(feature_names[i]) for i in indices][::-1]
    return list(zip(top_names, [round(float(v), 4) for v in top_importances])), importance_type
