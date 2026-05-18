"""
Fraud Detection Pipeline Package
Modular pipeline for multi-dataset fraud analysis.
"""

from pathlib import Path

PLOTS_DIR = Path("pipeline_plots")
PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = Path("pipeline_results.csv")
REPORT_DIR = Path("report")
REPORT_DIR.mkdir(exist_ok=True)
REPORT_ASSETS_DIR = REPORT_DIR / "report_assets"
REPORT_ASSETS_DIR.mkdir(exist_ok=True)

DATASETS = {
    "Credit Card Fraud": {"path": "creditcard.csv", "target": "Class"},
    "Online Payment Fraud": {"path": "onlinefraud.csv", "target": "isFraud"},
    "Bank Account Application Fraud": {"path": "Base.csv", "target": "fraud_bool"},
}

MODEL_COLORS = {
    "Dummy (most_frequent)": "#999999",
    "Dummy (stratified)":    "#b0b0b0",
    "Logistic Regression":   "#4c72b0",
    "Random Forest":         "#55a868",
    "XGBoost":               "#c44e52",
}
FRAUD_COLOR = "#d62728"
NON_FRAUD_COLOR = "#1f77b4"


def slug(name: str) -> str:
    """Convert a display name to a filename-safe slug."""
    return name.replace(" ", "_").replace("(", "").replace(")", "")
