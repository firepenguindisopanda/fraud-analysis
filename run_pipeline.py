"""
Fraud Detection Pipeline - Orchestrator
Run:  python run_pipeline.py
Produces: pipeline_results.csv, pipeline_plots/*.png, report data
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from pipeline import DATASETS, PLOTS_DIR, RESULTS_FILE
from pipeline.data_loader import load_dataset, run_quality_checks, clean_dataset, sample_dataset
from pipeline.eda import run_eda
from pipeline.preprocessing import prepare_data
from pipeline.modelling import train_and_evaluate, get_feature_importance, compute_confusion_matrix
from pipeline.visualizer import (
    plot_class_distribution, plot_correlations, plot_skewness,
    plot_feature_by_class, plot_categorical_fraud_rates,
    plot_confusion_matrix, plot_feature_importance, plot_summary_charts,
    plot_roc_curves, plot_pr_curves, plot_cost_heatmap,
    plot_radar_comparison, plot_business_impact,
    plot_temporal_fraud, plot_feature_importance_comparison,
    plot_threshold_sensitivity,
)

warnings.filterwarnings("ignore")


def process_dataset(dataset_name, info):
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*60}")

    df = load_dataset(info["path"])
    target_col = info["target"]

    if dataset_name == "Online Payment Fraud":
        df, original_size = sample_dataset(df, target_col, 200000)
        print(f"  Sampled to {len(df):,} rows (from {original_size:,})")
        df = df.drop(columns=["nameOrig", "nameDest"], errors="ignore")

    quality = run_quality_checks(df, target_col)
    print(f"  Shape: {quality['shape']}")
    print(f"  Fraud rate: {quality['fraud_rate']:.4%}")
    print(f"  Imbalance: {quality['imbalance_ratio']:.0f}:1")

    if dataset_name == "Bank Account Application Fraud":
        df, clean_info = clean_dataset(df)
        print(f"  Cleaned: {clean_info['duplicates_removed']} duplicates removed")

    print("\n  Running EDA...")
    numeric_features = ["Amount", "Time"] if dataset_name == "Credit Card Fraud" else None
    eda_results = run_eda(df, target_col, numeric_features)
    eda_output = {"dataset": dataset_name, "quality": quality, "eda": eda_results}
    eda_path = PLOTS_DIR / f"{dataset_name.replace(' ', '_')}_eda.json"
    with open(eda_path, "w") as f:
        json.dump(eda_output, f, indent=2, default=str)

    plot_class_distribution(df, target_col, dataset_name)
    plot_correlations(df, target_col, dataset_name)
    plot_skewness(df, target_col, dataset_name)

    if dataset_name == "Credit Card Fraud":
        for feat in ["Amount", "Time"]:
            plot_feature_by_class(df, target_col, dataset_name, feat)
        plot_temporal_fraud(df, target_col, dataset_name)
    elif dataset_name == "Online Payment Fraud":
        plot_feature_by_class(df, target_col, dataset_name, "amount")
        plot_categorical_fraud_rates(df, target_col, dataset_name)
    elif dataset_name == "Bank Account Application Fraud":
        for feat in ["income", "credit_risk_score", "session_length_in_minutes"]:
            if feat in df.columns:
                plot_feature_by_class(df, target_col, dataset_name, feat)
        plot_categorical_fraud_rates(df, target_col, dataset_name)

    print("\n  Preprocessing...")
    result = prepare_data(df, target_col)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, prep_info = result
    print(f"  Train: {prep_info['train_size']}, Test: {prep_info['test_size']}")
    print(f"  After SMOTE: {prep_info['train_class_distribution']}")

    print("\n  Training models...")
    results_df, fitted_models = train_and_evaluate(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, dataset_name)

    print("\n  Confusion matrices:")
    for model_name, obj in fitted_models.items():
        cm = confusion_matrix(y_test, obj["y_pred"])
        cm_info = compute_confusion_matrix(y_test, obj["y_pred"])
        plot_confusion_matrix(cm, dataset_name, model_name)
        print(f"    {model_name}: TP={cm_info['tp']}, FP={cm_info['fp']}, FN={cm_info['fn']}, TN={cm_info['tn']}")

    print("\n  Feature importance:")
    all_importances = {}
    for model_name, obj in fitted_models.items():
        importances, imp_type = get_feature_importance(obj["model"], X_train.columns)
        all_importances[model_name] = importances
        plot_feature_importance(importances, X_train.columns, dataset_name, model_name)

    print("\n  Generating ROC/PR curves...")
    plot_roc_curves(y_test, fitted_models, dataset_name)
    plot_pr_curves(y_test, fitted_models, dataset_name)

    metrics_for_radar = {}
    for _, row in results_df.iterrows():
        metrics_for_radar[row["model"]] = {"precision": row["precision"], "recall": row["recall"], "f1": row["f1"], "roc_auc": row["roc_auc"]}
    plot_radar_comparison(metrics_for_radar, dataset_name)

    print("  Generating business impact chart...")
    impact_data = {}
    for model_name, obj in fitted_models.items():
        cm_info = compute_confusion_matrix(y_test, obj["y_pred"])
        impact_data[model_name] = cm_info
    plot_business_impact(impact_data, dataset_name)

    print("  Generating feature importance comparison...")
    plot_feature_importance_comparison(all_importances, dataset_name)

    print("  Generating threshold sensitivity...")
    threshold_results = []
    for model_name, obj in fitted_models.items():
        if obj["y_prob"] is None:
            continue
        for thresh in np.arange(0.05, 0.95, 0.05):
            pred = (obj["y_prob"] >= thresh).astype(int)
            threshold_results.append({"model": model_name, "threshold": thresh, "precision": precision_score(y_test, pred, zero_division=0), "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0)})
    threshold_df = pd.DataFrame(threshold_results)
    plot_threshold_sensitivity(threshold_df, dataset_name)

    print("  Generating cost heatmap...")
    thresholds = np.arange(0.05, 0.95, 0.05)
    cost_ratios = [1, 5, 10, 50, 100, 500, 1000]
    best_model_name = results_df.iloc[0]["model"] if len(results_df) > 0 else "XGBoost"
    best_model_prob = fitted_models[best_model_name]["y_prob"] if best_model_name in fitted_models else None
    if best_model_prob is not None:
        cost_matrix = np.zeros((len(cost_ratios), len(thresholds)))
        for ri, ratio in enumerate(cost_ratios):
            for ti, thresh in enumerate(thresholds):
                pred = (best_model_prob >= thresh).astype(int)
                cm_info = compute_confusion_matrix(y_test, pred)
                cost_matrix[ri, ti] = ratio * cm_info["fn"] + 1 * cm_info["fp"]
        plot_cost_heatmap(cost_matrix, thresholds, cost_ratios, dataset_name)

    return results_df


def main():
    all_results_list = []
    for name, info in DATASETS.items():
        df = process_dataset(name, info)
        all_results_list.append(df)
    all_results = pd.concat(all_results_list, ignore_index=True)
    display_df = all_results.sort_values(["dataset", "f1"], ascending=[True, False])
    print(f"\n{'='*60}")
    print(f"  CROSS-DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"\n{display_df.to_string(index=False)}")
    plot_summary_charts(all_results)
    all_results.to_csv(RESULTS_FILE, index=False)
    print(f"\n  Results:   {RESULTS_FILE}")
    print(f"  Plots:     {PLOTS_DIR}/")
    print(f"\n  Run: python report_builder.py to generate HTML report")


if __name__ == "__main__":
    main()
