"""
Preprocess Base.csv (BAF NeurIPS 2022) into train/val/test numpy arrays.

Uses a temporal split by month — the correct evaluation strategy for fraud models:
  - Train : months 1–5
  - Val   : month 6
  - Test  : months 7–8

Usage:
    python DATA/preprocess.py           # default (no oversampling)
    python DATA/preprocess.py --smote   # SMOTE on training fold only
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "Base.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "processed")
RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
TARGET = "fraud_bool"


def load_and_split(smote: bool = False):
    if not os.path.exists(DATA_PATH):
        sys.exit(
            f"ERROR: {DATA_PATH} not found.\n"
            "Download Base.csv from https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 "
            "and place it in DATA/."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {df[TARGET].sum()} frauds "
          f"({df[TARGET].mean()*100:.3f}% fraud rate)")

    # Temporal split by month — preserves real-world ordering
    # months 1-5 → train, month 6 → val, months 7-8 → test
    train_df = df[df["month"] <= 5].copy()
    val_df   = df[df["month"] == 6].copy()
    test_df  = df[df["month"] >= 7].copy()

    print(f"\nTemporal split by month:")
    print(f"  Train (months 1-5) : {len(train_df):>7,}  fraud rate: {train_df[TARGET].mean():.4f}")
    print(f"  Val   (month 6)    : {len(val_df):>7,}  fraud rate: {val_df[TARGET].mean():.4f}")
    print(f"  Test  (months 7-8) : {len(test_df):>7,}  fraud rate: {test_df[TARGET].mean():.4f}")

    # Label-encode categorical features (fit on train only to prevent leakage)
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        val_df[col]   = le.transform(val_df[col].astype(str))
        test_df[col]  = le.transform(test_df[col].astype(str))
        encoders[col] = le

    feature_cols = [c for c in df.columns if c not in (TARGET, "month")]
    X_train = train_df[feature_cols].values
    X_val   = val_df[feature_cols].values
    X_test  = test_df[feature_cols].values
    y_train = train_df[TARGET].values
    y_val   = val_df[TARGET].values
    y_test  = test_df[TARGET].values

    # Scale numerical features (needed for logistic regression)
    numerical_cols = [c for c in feature_cols if c not in CATEGORICAL_FEATURES]
    numerical_idx  = [feature_cols.index(c) for c in numerical_cols]

    scaler = StandardScaler()
    X_train[:, numerical_idx] = scaler.fit_transform(X_train[:, numerical_idx])
    X_val[:, numerical_idx]   = scaler.transform(X_val[:, numerical_idx])
    X_test[:, numerical_idx]  = scaler.transform(X_test[:, numerical_idx])

    if smote:
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            sys.exit("ERROR: imbalanced-learn not installed. Run: pip install imbalanced-learn")
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"After SMOTE — train fraud rate: {y_train.mean():.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }
    for name, arr in splits.items():
        np.save(os.path.join(OUT_DIR, f"{name}.npy"), arr)

    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
    joblib.dump(encoders, os.path.join(OUT_DIR, "label_encoders.joblib"))
    joblib.dump(feature_cols, os.path.join(OUT_DIR, "feature_cols.joblib"))

    print(f"\nSaved arrays, scaler, and encoders to {OUT_DIR}")
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Base.csv (BAF NeurIPS 2022)")
    parser.add_argument(
        "--smote", action="store_true",
        help="Apply SMOTE oversampling to the training fold only"
    )
    args = parser.parse_args()
    load_and_split(smote=args.smote)
