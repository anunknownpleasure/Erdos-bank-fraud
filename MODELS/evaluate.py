"""
Evaluate all trained models on the held-out test set.

Requires:
  - outputs/processed/   (run DATA/preprocess.py first)
  - outputs/*.joblib     (run MODELS/train_models.py first)

Produces:
  - outputs/figures/pr_curves.png
  - outputs/figures/roc_curves.png
  - Console output with all KPIs

Usage:
    python MODELS/evaluate.py
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "best_thresholds.joblib")

MODEL_NAMES = ["LogisticRegression", "RandomForest", "LightGBM", "XGBoost"]


def load_test_data():
    if not os.path.exists(PROCESSED_DIR):
        sys.exit("ERROR: outputs/processed/ not found. Run: python DATA/preprocess.py")
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"), allow_pickle=True)
    return X_test, y_test


def load_models() -> dict:
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            print(f"WARNING: {path} not found, skipping {name}")
    if not models:
        sys.exit("ERROR: No trained models found. Run MODELS/train_models.ipynb first.")
    return models


def load_thresholds(model_names: list) -> dict:
    if os.path.exists(THRESHOLDS_PATH):
        return joblib.load(THRESHOLDS_PATH)
    print("WARNING: best_thresholds.joblib not found, defaulting to 0.5 for all models.")
    return {name: 0.5 for name in model_names}


def print_metrics(name: str, y_true, y_pred, y_proba, threshold: float):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr_val = fp / (fp + tn)
    fnr_val = fn / (fn + tp)
    auprc = average_precision_score(y_true, y_proba)

    print(f"\n{'='*55}")
    print(f"  {name}  (threshold={threshold:.4f})")
    print(f"{'='*55}")
    print(f"  AUPRC     : {auprc:.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_true, y_proba):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1        : {f1_score(y_true, y_pred):.4f}")
    print(f"  FPR       : {fpr_val:.4f}  ({fp} legitimate applications flagged)")
    print(f"  FNR       : {fnr_val:.4f}  ({fn} frauds missed)")
    print()
    print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"]))


def plot_pr_curves(models: dict, X_test, y_test):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = y_test.mean()
    ax.axhline(baseline, color="gray", linestyle="--", label=f"Baseline (prevalence={baseline:.4f})")
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, label=f"{name} (AUPRC={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Test Set")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "pr_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_roc_curves(models: dict, X_test, y_test):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "roc_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    X_test, y_test = load_test_data()
    models = load_models()
    thresholds = load_thresholds(list(models.keys()))

    print(f"\nTest set: {len(y_test):,} samples, {int(y_test.sum())} frauds "
          f"({y_test.mean()*100:.3f}% fraud rate)")

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        threshold = thresholds.get(name, 0.5)
        y_pred = (y_proba >= threshold).astype(int)
        print_metrics(name, y_test, y_pred, y_proba, threshold)

    plot_pr_curves(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
