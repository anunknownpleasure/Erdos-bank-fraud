"""
Preprocess creditcard.csv into train/val/test numpy arrays.

Usage:
    python DATA/preprocess.py           # class_weight approach (default)
    python DATA/preprocess.py --smote   # SMOTE oversampling on training fold
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "creditcard.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "processed")
RANDOM_STATE = 42


def load_and_split(smote: bool = False):
    if not os.path.exists(DATA_PATH):
        sys.exit(
            f"ERROR: {DATA_PATH} not found.\n"
            "Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in DATA/."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {df['Class'].sum()} frauds "
          f"({df['Class'].mean()*100:.3f}% fraud rate)")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Scale only the raw-magnitude features; V1-V28 are already PCA-normalized
    scaler = StandardScaler()
    X = X.copy()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    # Stratified 70 / 15 / 15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

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

    print(f"\nSplit sizes:")
    print(f"  Train : {len(y_train):>7,}  (fraud rate: {np.mean(y_train):.4f})")
    print(f"  Val   : {len(y_val):>7,}  (fraud rate: {np.mean(y_val):.4f})")
    print(f"  Test  : {len(y_test):>7,}  (fraud rate: {np.mean(y_test):.4f})")
    print(f"\nSaved arrays and scaler to {OUT_DIR}")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess creditcard.csv")
    parser.add_argument(
        "--smote", action="store_true",
        help="Apply SMOTE oversampling to the training fold only"
    )
    args = parser.parse_args()
    load_and_split(smote=args.smote)
