"""
Train and cross-validate fraud detection models on the BAF dataset.

Requires processed arrays in outputs/processed/ (run DATA/preprocess.py first).

Usage:
    python MODELS/train_models.py
"""

import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "processed")
MODEL_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
RANDOM_STATE = 42


def load_data() -> dict:
    if not os.path.exists(PROCESSED_DIR):
        sys.exit(
            f"ERROR: {PROCESSED_DIR} not found.\n"
            "Run: python DATA/preprocess.py"
        )
    arrays = {}
    for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        arrays[name] = np.load(
            os.path.join(PROCESSED_DIR, f"{name}.npy"), allow_pickle=True
        )
    return arrays


def build_models(scale_pos_weight: float) -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "LightGBM": lgb.LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=500, learning_rate=0.05,
            num_leaves=31, n_jobs=-1, random_state=RANDOM_STATE,
            verbose=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=500, learning_rate=0.05,
            max_depth=6, eval_metric="aucpr",
            n_jobs=-1, random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }


def train_and_select(data: dict) -> tuple:
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    neg, pos = np.bincount(y_train.astype(int))
    scale_pos_weight = neg / pos
    print(f"Training set — negatives: {neg:,}  positives: {pos:,}  "
          f"scale_pos_weight: {scale_pos_weight:.1f}")

    models = build_models(scale_pos_weight)
    # Use StratifiedKFold only within the training fold — temporal ordering
    # is already respected by the month-based split in preprocess.py
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n{'Model':<22} {'CV AUPRC (mean±std)':<26} {'Val AUPRC'}")
    print("-" * 65)

    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="average_precision", n_jobs=-1
        )
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        val_auprc = average_precision_score(y_val, val_proba)

        results[name] = {
            "model": model,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "val_auprc": val_auprc,
        }
        print(f"{name:<22} {cv_scores.mean():.4f} ± {cv_scores.std():.4f}        {val_auprc:.4f}")

    best_name = max(results, key=lambda k: results[k]["val_auprc"])
    print(f"\nBest model: {best_name}  (Val AUPRC: {results[best_name]['val_auprc']:.4f})")

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    for name, r in results.items():
        joblib.dump(r["model"], os.path.join(MODEL_OUT_DIR, f"{name}.joblib"))

    print(f"Saved all models to {MODEL_OUT_DIR}/")
    return results, best_name


if __name__ == "__main__":
    data = load_data()
    train_and_select(data)
