"""
Chart generation for fraud analysis pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from pipeline import PLOTS_DIR, MODEL_COLORS, FRAUD_COLOR, NON_FRAUD_COLOR, slug


def _save(fig, name):
    fig.savefig(PLOTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(df, target_col, dataset_name):
    counts = df[target_col].value_counts()
    total = counts.sum()
    pcts = counts / total
    classes = sorted(counts.index)
    labels = [f"Non-Fraud\n(Class 0)" if c == 0 else f"Fraud\n(Class {c})" for c in classes]
    bars = [counts[c] for c in classes]
    bar_colors = [NON_FRAUD_COLOR if c == 0 else FRAUD_COLOR for c in classes]
    fig, ax = plt.subplots(figsize=(6, 5))
    bar_containers = ax.bar(labels, bars, color=bar_colors, edgecolor="white", linewidth=1.2)
    for i, (bar, c) in enumerate(zip(bar_containers, bars)):
        pct = pcts[classes[i]]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{c:,}\n({pct:.2%})", ha="center", va="bottom", fontsize=11, fontweight="bold")
    imbalance = counts.get(0, 0) / max(counts.get(1, 1), 1)
    ax.text(0.5, 0.92, f"Imbalance ratio: {imbalance:.0f}:1  -  Fraud is {pcts.get(1, 0):.3%} of data", transform=ax.transAxes, ha="center", fontsize=10, bbox=dict(facecolor="yellow", alpha=0.3, boxstyle="round,pad=0.3"))
    ax.set_title(f"{dataset_name}\nFraud vs Non-Fraud Transaction Counts", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Transactions")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, f"{slug(dataset_name)}_class_dist.png")


def plot_correlations(df, target_col, dataset_name, top_n=10):
    numeric = df.select_dtypes(include=["number"]).columns
    if len(numeric) < 2:
        return
    corr = df[numeric].corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False).head(top_n)
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
        ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9, fontweight="bold", color=FRAUD_COLOR if v > 0 else NON_FRAUD_COLOR)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(range(len(vals_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.set_title(f"{dataset_name}\nTop {top_n} Feature Correlations with Target\n(Positive = more fraud when feature increases)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pearson Correlation")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.2f}"))
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, f"{slug(dataset_name)}_correlations.png")


def plot_skewness(df, target_col, dataset_name, top_n=8):
    numeric = df.select_dtypes(include=["number"]).columns
    skip = [target_col] if target_col in numeric else []
    skewed = df[numeric].skew().drop(skip, errors="ignore").abs().sort_values(ascending=False).head(top_n)
    if len(skewed) == 0:
        return
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.5)))
    vals = skewed.values[::-1]
    names = skewed.index[::-1]
    colors = [FRAUD_COLOR if v > 1 else NON_FRAUD_COLOR for v in vals]
    bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor="white")
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, f"{v:.1f}", va="center", fontsize=9, fontweight="bold")
    ax.axvline(1, color="black", linewidth=1.2, linestyle="--", alpha=0.6, label="Log-transform threshold (|skew| = 1)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(f"{dataset_name}\nMost Skewed Features", fontsize=12, fontweight="bold")
    ax.set_xlabel("Absolute Skewness")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, f"{slug(dataset_name)}_skew.png")


def plot_feature_by_class(df, target_col, dataset_name, feature_name):
    if feature_name not in df.columns:
        return
    classes = sorted(df[target_col].unique())
    subsets = {c: df[df[target_col] == c][feature_name].dropna() for c in classes}
    fraud_subset = subsets.get(1, pd.Series(dtype=float))
    nonfraud_subset = subsets.get(0, pd.Series(dtype=float))
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = min(80, int(np.sqrt(len(nonfraud_subset)))) if len(nonfraud_subset) > 0 else 50
    ax.hist(nonfraud_subset, bins=bins, density=True, alpha=0.5, color=NON_FRAUD_COLOR, label=f"Non-Fraud (n={len(nonfraud_subset):,})")
    ax.hist(fraud_subset, bins=bins, density=True, alpha=0.6, color=FRAUD_COLOR, label=f"Fraud (n={len(fraud_subset):,})")
    for c, color, ls in [(0, NON_FRAUD_COLOR, "--"), (1, FRAUD_COLOR, "-")]:
        s = subsets[c]
        if len(s) > 0:
            ax.axvline(s.mean(), color=color, linestyle=ls, linewidth=2, alpha=0.7, label=f"{'Non-Fraud' if c == 0 else 'Fraud'} mean={s.mean():.2f}")
    combined = pd.concat([nonfraud_subset, fraud_subset])
    x_upper = combined.quantile(0.995)
    ax.set_xlim(left=0, right=x_upper)
    ax.set_title(f"{dataset_name}\n{feature_name} Distribution by Class", fontsize=12, fontweight="bold")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, f"{slug(dataset_name)}_{slug(feature_name)}_by_class.png")


def plot_categorical_fraud_rates(df, target_col, dataset_name, top_n=10):
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    overall_rate = df[target_col].mean()
    for col in cat_cols[:3]:
        rates = df.groupby(col)[target_col].mean().sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.45)))
        vals = rates.values[::-1]
        names = [str(n) for n in rates.index[::-1]]
        colors = [FRAUD_COLOR if v > overall_rate else NON_FRAUD_COLOR for v in vals]
        bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor="white")
        for i, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, f"{v:.2%}", va="center", fontsize=9, fontweight="bold")
        ax.axvline(overall_rate, color="black", linewidth=1.2, linestyle="--", alpha=0.7, label=f"Overall fraud rate: {overall_rate:.3%}")
        ax.legend(fontsize=9, loc="lower right")
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title(f"{dataset_name}\nFraud Rate by {col} (Top {top_n})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Fraud Rate")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.spines[["top", "right"]].set_visible(False)
        _save(fig, f"{slug(dataset_name)}_{col}_fraud_rate.png")


def plot_confusion_matrix(cm, dataset_name, model_name):
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    cm_pct = cm / total * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNon-Fraud (0)", "Predicted\nFraud (1)"], fontsize=10)
    ax.set_yticklabels(["Actual\nNon-Fraud (0)", "Actual\nFraud (1)"], fontsize=10)
    quad_labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            txt = f"{quad_labels[i][j]}\n{cm[i,j]:,}\n({cm_pct[i,j]:.2%})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=13, fontweight="bold", color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_title(f"{dataset_name} - {model_name}\nConfusion Matrix", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"{slug(dataset_name)}_{slug(model_name)}_cm.png")


def plot_feature_importance(importances, feature_names, dataset_name, model_name, top_n=15):
    if not importances:
        return
    names, values = zip(*importances[:top_n])
    names = names[::-1]
    values = values[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, min(top_n, len(names)) * 0.45)))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))[::-1]
    bars = ax.barh(range(len(values)), values, color=colors, edgecolor="white")
    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(bar.get_width() + max(values) * 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(f"{dataset_name} - {model_name}\nTop {min(top_n, len(names))} Features", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, f"{slug(dataset_name)}_{slug(model_name)}_importance.png")


def plot_summary_charts(all_results):
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        pivot = all_results.pivot(index="dataset", columns="model", values=metric)
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
            bars = ax.bar(x + offset, vals, bar_width, label=model_name, color=color, edgecolor="white", alpha=0.9)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=0)
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("Fraud", "Frd") for d in datasets], fontsize=10)
        ax.set_title(f"Model Comparison by {metric.upper()}\n(across all 3 datasets)", fontsize=13, fontweight="bold")
        ax.set_ylabel(metric.upper())
        ax.legend(title="Model", fontsize=8, title_fontsize=9, loc="best")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, min(1.0, max(pivot.max()) * 1.25))
        _save(fig, f"summary_{metric}.png")


def plot_roc_curves(y_test, models_dict, dataset_name):
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, obj in models_dict.items():
        if obj["y_prob"] is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, obj["y_prob"])
        roc_auc = auc(fpr, tpr)
        color = MODEL_COLORS.get(model_name, "#333333")
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"{dataset_name}\nROC Curves - Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, f"{slug(dataset_name)}_roc_curves.png")


def plot_pr_curves(y_test, models_dict, dataset_name):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, obj in models_dict.items():
        if obj["y_prob"] is None:
            continue
        precision, recall, _ = precision_recall_curve(y_test, obj["y_prob"])
        avg_pr = average_precision_score(y_test, obj["y_prob"])
        color = MODEL_COLORS.get(model_name, "#333333")
        ax.plot(recall, precision, color=color, linewidth=2, label=f"{model_name} (AP = {avg_pr:.3f})")
    fraud_rate = y_test.mean()
    ax.axhline(fraud_rate, color="gray", linestyle=":", linewidth=1, alpha=0.5, label=f"No-skill (PR = {fraud_rate:.3f})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"{dataset_name}\nPrecision-Recall Curves - Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    _save(fig, f"{slug(dataset_name)}_pr_curves.png")


def plot_cost_heatmap(cost_matrix, thresholds, cost_ratios, dataset_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(cost_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(thresholds)))
    ax.set_yticks(range(len(cost_ratios)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels([f"1:{r}" for r in cost_ratios], fontsize=9)
    for i in range(len(cost_ratios)):
        for j in range(len(thresholds)):
            val = cost_matrix[i, j]
            ax.text(j, i, f"${val:,.0f}", ha="center", va="center", fontsize=7, color="white" if val > cost_matrix.max() * 0.5 else "black")
    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Cost Ratio (FN:FP)", fontsize=11)
    ax.set_title(f"{dataset_name}\nTotal Cost Heatmap (Threshold x Cost Ratio)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Total Cost ($)")
    fig.tight_layout()
    _save(fig, f"{slug(dataset_name)}_cost_heatmap.png")


def plot_radar_comparison(metrics_dict, dataset_name):
    categories = ["Precision", "Recall", "F1", "ROC-AUC"]
    N = len(categories)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for model_name, metrics in metrics_dict.items():
        values = [metrics.get("precision", 0), metrics.get("recall", 0), metrics.get("f1", 0), metrics.get("roc_auc", 0)]
        values += values[:1]
        color = MODEL_COLORS.get(model_name, "#333333")
        ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f"{dataset_name}\nModel Performance Radar", fontsize=13, fontweight="bold", pad=20)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save(fig, f"{slug(dataset_name)}_radar.png")


def plot_business_impact(results_dict, dataset_name, fn_cost=100, fp_cost=1):
    models = list(results_dict.keys())
    fraud_caught = [r["tp"] * fn_cost for r in results_dict.values()]
    false_alarms = [r["fp"] * fp_cost for r in results_dict.values()]
    missed_fraud = [r["fn"] * fn_cost for r in results_dict.values()]
    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, fraud_caught, width, label="Fraud Caught (value saved)", color="#2ecc71", edgecolor="white")
    bars2 = ax.bar(x, false_alarms, width, label="False Alarms (cost)", color="#f39c12", edgecolor="white")
    bars3 = ax.bar(x + width, missed_fraud, width, label="Missed Fraud (cost)", color="#e74c3c", edgecolor="white")
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height, f"${height:,.0f}", ha="center", va="bottom", fontsize=7, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (", "\n(") for m in models], fontsize=8)
    ax.set_ylabel("Cost ($)", fontsize=11)
    ax.set_title(f"{dataset_name}\nBusiness Impact Analysis\n(FN=${fn_cost}, FP=${fp_cost})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _save(fig, f"{slug(dataset_name)}_business_impact.png")


def plot_temporal_fraud(df, target_col, dataset_name, time_col="Time"):
    if time_col not in df.columns:
        return
    df_temp = df.copy()
    df_temp["hour"] = (df_temp[time_col] / 3600).astype(int) % 24
    hourly = df_temp.groupby("hour")[target_col].agg(["mean", "count"])
    hourly.columns = ["fraud_rate", "count"]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color1 = FRAUD_COLOR
    ax1.set_xlabel("Hour of Day", fontsize=11)
    ax1.set_ylabel("Fraud Rate (%)", color=color1, fontsize=11)
    ax1.plot(hourly.index, hourly["fraud_rate"] * 100, color=color1, linewidth=2, marker="o", markersize=4)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_title(f"{dataset_name}\nFraud Rate by Hour of Day", fontsize=13, fontweight="bold")
    ax2 = ax1.twinx()
    color2 = NON_FRAUD_COLOR
    ax2.set_ylabel("Transaction Count", color=color2, fontsize=11)
    ax2.bar(hourly.index, hourly["count"], color=color2, alpha=0.3)
    ax2.tick_params(axis="y", labelcolor=color2)
    fig.tight_layout()
    _save(fig, f"{slug(dataset_name)}_temporal_fraud.png")


def plot_feature_importance_comparison(all_importances, dataset_name):
    models = list(all_importances.keys())
    all_features = set()
    for imp_list in all_importances.values():
        for name, _ in imp_list:
            all_features.add(name)
    top_features = list(all_features)[:10]
    x = np.arange(len(top_features))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 5))
    for mi, model_name in enumerate(models):
        imp_dict = {name: val for name, val in all_importances[model_name]}
        vals = [imp_dict.get(f, 0) for f in top_features]
        color = MODEL_COLORS.get(model_name, "#333333")
        offset = (mi - (len(models) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=model_name, color=color, edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(top_features, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Importance", fontsize=11)
    ax.set_title(f"{dataset_name}\nFeature Importance Comparison Across Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, f"{slug(dataset_name)}_feature_comparison.png")


def plot_threshold_sensitivity(threshold_results, dataset_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for model_name, color in MODEL_COLORS.items():
        if model_name.startswith("Dummy"):
            continue
        model_data = threshold_results[threshold_results["model"] == model_name]
        if model_data.empty:
            continue
        axes[0].plot(model_data["threshold"], model_data["precision"], color=color, linewidth=2, label=model_name)
        axes[1].plot(model_data["threshold"], model_data["recall"], color=color, linewidth=2, label=model_name)
        axes[2].plot(model_data["threshold"], model_data["f1"], color=color, linewidth=2, label=model_name)
    for ax, title in [(axes[0], "Precision"), (axes[1], "Recall"), (axes[2], "F1 Score")]:
        ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5, label="Default (0.5)")
        ax.set_xlabel("Threshold", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"{dataset_name}\n{title} vs Threshold", fontsize=12, fontweight="bold")
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    _save(fig, f"{slug(dataset_name)}_threshold_sensitivity.png")
