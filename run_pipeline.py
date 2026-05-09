"""
Fraud Detection Pipeline
Run:  python run_pipeline.py
Produces: results.csv, plots/*.png, trained models
"""

import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

PLOTS_DIR = Path("pipeline_plots")
PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = "pipeline_results.csv"

# Dataset config
DATASETS = {
    "Credit Card Fraud": {
        "path": "creditcard.csv",
        "target": "Class"
    },
    "Online Payment Fraud": {
        "path": "onlinefraud.csv",
        "target": "isFraud"
    },
    "Bank Account Application Fraud": {
        "path": "Base.csv",
        "target": "fraud_bool"
    }
}

# Color palette
MODEL_COLORS = {
    "Dummy (most_frequent)": "#999999",
    "Dummy (stratified)":    "#b0b0b0",
    "Logistic Regression":   "#4c72b0",
    "Random Forest":         "#55a868",
    "XGBoost":               "#c44e52",
}
FRAUD_COLOR   = "#d62728"
NON_FRAUD_COLOR = "#1f77b4"


def load_dataset(path):
    return pd.read_csv(path)


def basic_overview(df, dataset_name, target_col):
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Target distribution:\n    {df[target_col].value_counts().to_string().replace(chr(10), chr(10) + '    ')}")


# Plotting helpers

def _slug(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def _fmt_pct(x):
    return f"{x:.1%}"


def _save(fig, name):
    fig.savefig(PLOTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(df, target_col, dataset_name):
    counts = df[target_col].value_counts()
    total = counts.sum()
    pcts = counts / total

    print(f"\n  [Class Distribution - {dataset_name}]")
    print(f"    Class 0 (non-fraud): {counts.get(0, 0):>10,}  ({pcts.get(0, 0):.2%})")
    print(f"    Class 1 (fraud):     {counts.get(1, 0):>10,}  ({pcts.get(1, 0):.2%})")
    print(f"    Imbalance ratio:     {counts.get(0, 0) / max(counts.get(1, 1), 1):>.1f}:1")

    classes = sorted(counts.index)
    labels = [f"Non-Fraud\n(Class 0)" if c == 0 else f"Fraud\n(Class {c})" for c in classes]
    bars = [counts[c] for c in classes]
    bar_colors = [NON_FRAUD_COLOR if c == 0 else FRAUD_COLOR for c in classes]

    fig, ax = plt.subplots(figsize=(6, 5))
    bar_containers = ax.bar(labels, bars, color=bar_colors, edgecolor="white", linewidth=1.2)

    for i, (bar, c) in enumerate(zip(bar_containers, bars)):
        pct = pcts[classes[i]]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{c:,}\n({pct:.2%})", ha="center", va="bottom", fontsize=11, fontweight="bold")

    imbalance = counts[0] / max(counts.get(1, 1), 1)
    ax.text(0.5, 0.92, f"Imbalance ratio: {imbalance:.0f}:1  -  Fraud is {pcts.get(1, 0):.3%} of data",
            transform=ax.transAxes, ha="center", fontsize=10,
            bbox=dict(facecolor="yellow", alpha=0.3, boxstyle="round,pad=0.3"))

    ax.set_title(f"{dataset_name}\nFraud vs Non-Fraud Transaction Counts", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Transactions")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)

    _save(fig, f"{_slug(dataset_name)}_class_dist.png")


def plot_top_numeric_correlations(df, target_col, dataset_name, top_n=10):
    numeric = df.select_dtypes(include=["number"]).columns
    if len(numeric) < 2:
        return

    corr = df[numeric].corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False).head(top_n)

    print(f"\n  [Top {top_n} Correlations with Target - {dataset_name}]")
    for feat, val in corr.items():
        print(f"    {feat:40s}  {val:>8.4f}")

    colors = [FRAUD_COLOR if v > 0 else NON_FRAUD_COLOR for v in corr.values]
    vals_sorted = corr.values[::-1]
    names_sorted = corr.index[::-1]
    colors_sorted = colors[::-1]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.45)))
    bars = ax.barh(range(len(vals_sorted)), vals_sorted, color=colors_sorted, edgecolor="white")

    for i, (bar, v) in enumerate(zip(bars, vals_sorted)):
        label = f"{v:+.4f}"
        x_pos = bar.get_width()
        offset = 0.005 if v >= 0 else -0.04
        ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                label, va="center",
                fontsize=9, fontweight="bold",
                color=FRAUD_COLOR if v > 0 else NON_FRAUD_COLOR)

    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(range(len(vals_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.set_title(f"{dataset_name}\nTop {top_n} Feature Correlations with Target\n(Positive = more fraud when feature ↑)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pearson Correlation")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.2f}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    _save(fig, f"{_slug(dataset_name)}_correlations.png")


def plot_top_skewed_features(df, target_col, dataset_name, top_n=8):
    numeric = df.select_dtypes(include=["number"]).columns
    skip = [target_col] if target_col in numeric else []
    skewed = df[numeric].skew().drop(skip, errors="ignore").abs().sort_values(ascending=False).head(top_n)

    if len(skewed) == 0:
        return

    print(f"\n  [Top {top_n} Skewed Features - {dataset_name}]")
    print(f"    (|Skew| > 1 = log-transform applied; higher = more extreme tail)")
    for feat, val in skewed.items():
        flag = " ← log-transformed" if val > 1 else ""
        print(f"    {feat:40s}  |skew| = {val:>8.2f}{flag}")

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.5)))
    vals = skewed.values[::-1]
    names = skewed.index[::-1]
    colors = [FRAUD_COLOR if v > 1 else NON_FRAUD_COLOR for v in vals]

    bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor="white")

    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=9, fontweight="bold")

    ax.axvline(1, color="black", linewidth=1.2, linestyle="--", alpha=0.6,
               label="Log-transform threshold (|skew| = 1)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(f"{dataset_name}\nMost Skewed Features\n(Red = above log-transform threshold)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Absolute Skewness")
    ax.spines[["top", "right"]].set_visible(False)

    _save(fig, f"{_slug(dataset_name)}_skew.png")


def plot_numeric_feature_by_class(df, target_col, dataset_name, feature_name):
    if feature_name not in df.columns:
        return

    classes = sorted(df[target_col].unique())
    subsets = {c: df[df[target_col] == c][feature_name].dropna() for c in classes}
    fraud_subset = subsets.get(1, pd.Series(dtype=float))
    nonfraud_subset = subsets.get(0, pd.Series(dtype=float))

    print(f"\n  [Feature: {feature_name} - {dataset_name}]")
    for c in classes:
        s = subsets[c]
        label = "Fraud" if c == 1 else "Non-Fraud"
        print(f"    {label:12s}  n={len(s):>8,}  mean={s.mean():>10.2f}  median={s.median():>10.2f}  std={s.std():>10.2f}")
    if len(fraud_subset) > 0 and len(nonfraud_subset) > 0:
        from scipy.stats import ks_2samp
        ks_stat, ks_p = ks_2samp(nonfraud_subset, fraud_subset)
        print(f"    KS test:     statistic={ks_stat:.4f}, p-value={ks_p:.2e} "
              + ("(distributions differ significantly)" if ks_p < 0.05 else "(no significant difference)"))

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = min(80, int(np.sqrt(len(nonfraud_subset))))

    ax.hist(nonfraud_subset, bins=bins, density=True, alpha=0.5,
            color=NON_FRAUD_COLOR, label=f"Non-Fraud (n={len(nonfraud_subset):,})")
    ax.hist(fraud_subset, bins=bins, density=True, alpha=0.6,
            color=FRAUD_COLOR, label=f"Fraud (n={len(fraud_subset):,})")

    for c, color, ls in [(0, NON_FRAUD_COLOR, "--"), (1, FRAUD_COLOR, "-")]:
        s = subsets[c]
        if len(s) > 0:
            ax.axvline(s.mean(), color=color, linestyle=ls, linewidth=2,
                       alpha=0.7, label=f"{'Non-Fraud' if c == 0 else 'Fraud'} mean={s.mean():.2f}")

    # Focus x-axis on the main distribution (99.5th percentile)
    combined = pd.concat([nonfraud_subset, fraud_subset])
    x_upper = combined.quantile(0.995)
    ax.set_xlim(left=0, right=x_upper)
    outliers_beyond = (combined > x_upper).sum()
    if outliers_beyond > 0:
        ax.text(0.98, 0.95, f"{outliers_beyond} extreme points\nbeyond ${x_upper:,.0f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                style="italic", color="gray")

    ax.set_title(f"{dataset_name}\n{feature_name} Distribution by Class", fontsize=12, fontweight="bold")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="best")
    ax.spines[["top", "right"]].set_visible(False)

    _save(fig, f"{_slug(dataset_name)}_{feature_name}_by_class.png")


def show_top_categorical_fraud_rates(df, target_col, dataset_name, top_n=10):
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    overall_rate = df[target_col].mean()

    for col in cat_cols[:3]:
        rates = df.groupby(col)[target_col].mean().sort_values(ascending=False).head(top_n)
        counts = df.groupby(col)[target_col].count()

        print(f"\n  [Fraud Rate by {col} - {dataset_name}]")
        print(f"    Overall fraud rate: {overall_rate:.4%}")
        for cat, rate in rates.items():
            n = counts[cat]
            print(f"    {str(cat):30s}  rate={rate:>7.2%}  (n={n:,})")

        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.45)))
        vals = rates.values[::-1]
        names = [str(n) for n in rates.index[::-1]]
        colors = [FRAUD_COLOR if v > overall_rate else NON_FRAUD_COLOR for v in vals]

        bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor="white")

        for i, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2%}", va="center", fontsize=9, fontweight="bold")

        ax.axvline(overall_rate, color="black", linewidth=1.2, linestyle="--",
                   alpha=0.7, label=f"Overall fraud rate: {overall_rate:.3%}")
        ax.legend(fontsize=9, loc="lower right")
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title(f"{dataset_name}\nFraud Rate by {col} (Top {top_n})\n(Red = above overall average)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Fraud Rate")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.spines[["top", "right"]].set_visible(False)

        _save(fig, f"{_slug(dataset_name)}_{col}_fraud_rate.png")


def preprocess_for_model(df, target_col):
    df = df.copy()
    X = df.drop(columns=[target_col])
    y = df[target_col]

    for col in X.select_dtypes(include=["number"]).columns:
        if abs(X[col].skew()) > 1:
            X[col] = np.log1p(X[col])
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    X = pd.get_dummies(X, drop_first=True)

    for col in X.columns:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Original training distribution: {dict(Counter(y_train))}")

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {dict(Counter(y_train_smote))}")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_smote), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    return X_train_smote, X_test, X_train_scaled, X_test_scaled, y_train_smote, y_test


def get_models():
    models = {}
    models["Dummy (most_frequent)"] = DummyClassifier(strategy="most_frequent", random_state=42)
    models["Dummy (stratified)"] = DummyClassifier(strategy="stratified", random_state=42)
    models["Logistic Regression"] = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
    )
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0
    )
    return models


def train_and_evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, dataset_name):
    models = get_models()
    rows = []
    fitted_models = {}

    for model_name, model in models.items():
        if "Dummy" in model_name:
            train_X, test_X = X_train, X_test
        elif model_name == "Logistic Regression":
            train_X, test_X = X_train_scaled, X_test_scaled
        else:
            train_X, test_X = X_train, X_test

        model.fit(train_X, y_train)
        y_pred = model.predict(test_X)
        y_prob = model.predict_proba(test_X)[:, 1] if hasattr(model, "predict_proba") else None

        rows.append({
            "dataset": dataset_name,
            "model": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
        })
        fitted_models[model_name] = {"model": model, "y_pred": y_pred, "y_prob": y_prob}

    return pd.DataFrame(rows).sort_values("f1", ascending=False), fitted_models


def plot_confusion_matrices(fitted_models, X_test, X_test_scaled, y_test, dataset_name):
    for model_name, obj in fitted_models.items():
        if "Dummy" in model_name:
            test_X = X_test
        elif model_name == "Logistic Regression":
            test_X = X_test_scaled
        else:
            test_X = X_test

        cm = confusion_matrix(y_test, obj["y_pred"])
        tn, fp, fn, tp = cm.ravel()

        print(f"\n  [Confusion Matrix - {dataset_name} / {model_name}]")
        print(f"    TN={tn:,}  FP={fp:,}")
        print(f"    FN={fn:,}  TP={tp:,}")
        print(f"    Sensitivity (TPR):     {tp/max(tp+fn, 1):.2%}")
        print(f"    Specificity (TNR):     {tn/max(tn+fp, 1):.2%}")
        print(f"    Precision (PPV):       {tp/max(tp+fp, 1):.2%}")
        print(f"    False Positive Rate:   {fp/max(fp+tn, 1):.2%}")

        total = cm.sum()
        cm_pct = cm / total * 100

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nNon-Fraud (0)", "Predicted\nFraud (1)"], fontsize=10)
        ax.set_yticklabels(["Actual\nNon-Fraud (0)", "Actual\nFraud (1)"], fontsize=10)

        quad_labels = [["TN", "FP"], ["FN", "TP"]]
        colors = [["white", "white"], ["white", "white"]]
        for i in range(2):
            for j in range(2):
                txt = f"{quad_labels[i][j]}\n{cm[i,j]:,}\n({cm_pct[i,j]:.2%})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=13,
                        fontweight="bold", color="white" if cm[i, j] > cm.max() * 0.6 else "black")

        ax.set_title(f"{dataset_name} - {model_name}\nConfusion Matrix", fontsize=12, fontweight="bold")
        fig.tight_layout()
        _save(fig, f"{_slug(dataset_name)}_{_slug(model_name)}_cm.png")


def plot_feature_importance(model, feature_names, dataset_name, model_name, top_n=15):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_type = "Gini importance"
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1 and coef.shape[0] == 1:
            coef = coef[0]
        importances = np.abs(coef)
        importance_type = "|Coefficient|"
    else:
        return

    if len(importances) != len(feature_names):
        print(f"    Skipping feature importance for {model_name} - shape mismatch")
        return

    indices = np.argsort(importances)[-top_n:]
    top_importances = importances[indices][::-1]
    top_names = [feature_names[i] for i in indices][::-1]

    print(f"\n  [Top {top_n} Features - {dataset_name} / {model_name}]")
    for name, val in zip(top_names, top_importances):
        print(f"    {name:45s}  {val:.4f}")

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.45)))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_names)))[::-1]
    bars = ax.barh(range(len(top_importances)), top_importances, color=colors, edgecolor="white")

    for i, (bar, v) in enumerate(zip(bars, top_importances)):
        ax.text(bar.get_width() + max(top_importances) * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(range(len(top_importances)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_title(f"{dataset_name} - {model_name}\nTop {top_n} Features ({importance_type})", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"{importance_type}")
    ax.spines[["top", "right"]].set_visible(False)

    _save(fig, f"{_slug(dataset_name)}_{_slug(model_name)}_importance.png")


def run_data_quality_checks(df, target_col, duplicate_rows, infinite_values, constant_cols, near_constant_cols):
    summary_df = pd.DataFrame({
        "Issue": ["Duplicate Rows", "Infinite Values", "Constant Columns", "Near-Constant Columns"],
        "Count": [duplicate_rows, infinite_values, len(constant_cols), len(near_constant_cols)]
    })
    print(f"\n  Data Quality Summary:\n{summary_df.to_string()}")
    print(f"  Shape: {df.shape}")
    class_counts = df[target_col].value_counts()
    imbalance_ratio = class_counts[0] / class_counts[1]
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
    print(f"  No-skill baseline accuracy: {class_counts.max() / class_counts.sum():.4f}")
    weights = compute_class_weight("balanced", classes=np.unique(df[target_col]), y=df[target_col])
    print(f"  Class weights: {dict(zip(np.unique(df[target_col]), weights))}")


def clean_dataset(df):
    df = df.copy()
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"  Duplicate rows removed: {before - df.shape[0]}")

    df = df.replace([np.inf, -np.inf], np.nan)
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df = df.drop(columns=const_cols)
    if const_cols:
        print(f"  Constant columns removed: {const_cols}")

    dup_cols = []
    for i in range(df.shape[1]):
        for j in range(i + 1, df.shape[1]):
            if df.iloc[:, i].equals(df.iloc[:, j]):
                dup_cols.append(df.columns[j])
    df = df.drop(columns=dup_cols)
    if dup_cols:
        print(f"  Duplicated columns removed: {dup_cols}")

    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(exclude=["number"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def plot_summary_charts(all_results):
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"\n  [Summary - {metric.upper()}]")
        pivot = all_results.pivot(index="dataset", columns="model", values=metric)
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

        fig, ax = plt.subplots(figsize=(11, 5.5))
        datasets = pivot.index.tolist()
        models_list = pivot.columns.tolist()
        x = np.arange(len(datasets))
        n_models = len(models_list)
        bar_width = 0.8 / n_models

        for mi, model_name in enumerate(models_list):
            offset = (mi - (n_models - 1) / 2) * bar_width
            vals = pivot[model_name].values
            color = MODEL_COLORS.get(model_name, "#333333")
            bars = ax.bar(x + offset, vals, bar_width, label=model_name,
                          color=color, edgecolor="white", alpha=0.9)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=0)

        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("Fraud", "Frd") for d in datasets], fontsize=10)
        ax.set_title(f"Model Comparison by {metric.upper()}\n(across all 3 datasets)", fontsize=13, fontweight="bold")
        ax.set_ylabel(metric.upper())
        ax.legend(title="Model", fontsize=8, title_fontsize=9, loc="best")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, min(1.0, max(pivot.max()) * 1.25))

        _save(fig, f"summary_{metric}.png")

    all_results.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved to {RESULTS_FILE}")


def process_dataset(dataset_name, info):
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*60}")

    df = load_dataset(info["path"])
    target_col = info["target"]

    if dataset_name == "Online Payment Fraud":
        df, _ = train_test_split(
            df, train_size=200000,
            stratify=df[target_col], random_state=42
        )
        print(f"  Sampled to {len(df):,} rows (stratified)")
        df = df.drop(columns=["nameOrig", "nameDest"], errors="ignore")

    print("\nBasic Overview")
    basic_overview(df, dataset_name, target_col)

    plot_class_distribution(df, target_col, dataset_name)
    plot_top_numeric_correlations(df, target_col, dataset_name, top_n=10)
    plot_top_skewed_features(df, target_col, dataset_name, top_n=8)

    if dataset_name == "Credit Card Fraud":
        for feat in ["Amount", "Time"]:
            plot_numeric_feature_by_class(df, target_col, dataset_name, feat)
    elif dataset_name == "Online Payment Fraud":
        plot_numeric_feature_by_class(df, target_col, dataset_name, "amount")
        show_top_categorical_fraud_rates(df, target_col, dataset_name, top_n=10)
    elif dataset_name == "Bank Account Application Fraud":
        duplicate_rows = df.duplicated().sum()
        infinite_values = np.isinf(df.select_dtypes(include=["number"])).sum().sum()
        constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
        near_constant_cols = [c for c in df.columns if df[c].nunique() < 5]
        run_data_quality_checks(df, target_col, duplicate_rows, infinite_values, constant_cols, near_constant_cols)
        df = clean_dataset(df)
        print("\nCleaned Dataset")
        print(f"  Shape after cleaning: {df.shape}")
        for feature in ["income", "credit_risk_score", "session_length_in_minutes", "device_fraud_count"]:
            plot_numeric_feature_by_class(df, target_col, dataset_name, feature)
        show_top_categorical_fraud_rates(df, target_col, dataset_name, top_n=10)

    print("\nPreprocessing")
    result = preprocess_for_model(df, target_col)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = result

    print("\nTraining & Evaluation")
    results_df, fitted_models = train_and_evaluate_models(
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, dataset_name
    )
    print(f"\n{results_df.to_string(index=False)}")

    print("\nConfusion Matrices")
    plot_confusion_matrices(fitted_models, X_test, X_test_scaled, y_test, dataset_name)

    print("\nFeature Importance")
    for model_name, obj in fitted_models.items():
        plot_feature_importance(obj["model"], X_train.columns, dataset_name, model_name, top_n=15)

    return results_df


def main():
    all_results_list = []

    for name, info in DATASETS.items():
        df = process_dataset(name, info)
        all_results_list.append(df)

    all_results = pd.concat(all_results_list, ignore_index=True)
    display_df = all_results.sort_values(["dataset", "f1"], ascending=[True, False])
    print(f"\n{display_df.to_string(index=False)}")

    plot_summary_charts(all_results)

    print(f"  Results:   {RESULTS_FILE}")
    print(f"  Plots:     {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
