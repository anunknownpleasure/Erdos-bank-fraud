"""
Preprocess Base.csv (BAF NeurIPS 2022) into train/val/test numpy arrays.

Uses a temporal split by month — the correct evaluation strategy for fraud models:
  - Train : months 1–5
  - Val   : month 6
  - Test  : months 7–8

Feature selection mirrors the methodology in EDA/fraud_eda.ipynb: numerical features
are ranked by absolute point-biserial correlation with fraud_bool (computed on training
data only) and the top TOP_N_NUMERICAL are kept. All 5 categorical features are always
included, consistent with the categorical fraud-rate analysis in both EDA notebooks.

EDA/Bank_fraud_EDA.ipynb explicitly identified income, customer_age, credit_risk_score,
and keep_alive_session as the most visually discriminative numerical features; these
will be captured within the top-N selection.

Usage:
    python DATA/preprocess.py                  # with feature selection (default)
    python DATA/preprocess.py --no-select      # skip feature selection, use all features
    python DATA/preprocess.py --smote          # SMOTE on training fold only
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

# Keep the top N numerical features by |correlation with fraud_bool|.
# EDA/fraud_eda.ipynb visualises the top 12 for KDE plots and top 8 for boxplots;
# 15 is chosen to cover those ranges while avoiding noise from weakly correlated features.
TOP_N_NUMERICAL = 15


def select_top_corr_features(train_df, y_train, numerical_cols):
    """
    Return the TOP_N_NUMERICAL features with highest |correlation with fraud_bool|,
    fitted on training data only (mirrors fraud_eda.ipynb's corr_with_target ranking).
    """
    corr = train_df[numerical_cols].corrwith(y_train).abs().sort_values(ascending=False)
    kept = corr.head(TOP_N_NUMERICAL).index.tolist()

    print(f"\nFeature selection — top {TOP_N_NUMERICAL} numerical by |corr with {TARGET}|:")
    print(f"  {'Feature':<40} {'|corr|':>8}")
    print(f"  {'-'*50}")
    for feat in kept:
        print(f"  {feat:<40} {corr[feat]:>8.4f}")
    dropped = [c for c in numerical_cols if c not in kept]
    if dropped:
        print(f"\n  Dropped ({len(dropped)}): {dropped}")

    return kept


def load_and_split(smote: bool = False, feature_selection: bool = True):
    if not os.path.exists(DATA_PATH):
        sys.exit(
            f"ERROR: {DATA_PATH} not found.\n"
            "Download Base.csv from https://www.kaggle.com/datasets/sgpjesus/"
            "bank-account-fraud-dataset-neurips-2022 and place it in DATA/."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {df[TARGET].sum()} frauds "
          f"({df[TARGET].mean()*100:.3f}% fraud rate)")

    # Temporal split by month
    train_df = df[df["month"] <= 5].copy()
    val_df   = df[df["month"] == 6].copy()
    test_df  = df[df["month"] >= 7].copy()

    print(f"\nTemporal split by month:")
    print(f"  Train (months 1-5) : {len(train_df):>7,}  fraud rate: {train_df[TARGET].mean():.4f}")
    print(f"  Val   (month 6)    : {len(val_df):>7,}  fraud rate: {val_df[TARGET].mean():.4f}")
    print(f"  Test  (months 7-8) : {len(test_df):>7,}  fraud rate: {test_df[TARGET].mean():.4f}")

    # Label-encode categorical features (fit on train only)
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        val_df[col]   = le.transform(val_df[col].astype(str))
        test_df[col]  = le.transform(test_df[col].astype(str))
        encoders[col] = le

    all_feature_cols = [c for c in df.columns if c not in (TARGET, "month")]
    numerical_cols   = [c for c in all_feature_cols if c not in CATEGORICAL_FEATURES]

    # Feature selection (fitted on training data only)
    if feature_selection:
        kept_numerical = select_top_corr_features(
            train_df[numerical_cols], train_df[TARGET], numerical_cols
        )
    else:
        kept_numerical = numerical_cols
        print(f"\nFeature selection skipped — using all {len(numerical_cols)} numerical features.")

    feature_cols = kept_numerical + CATEGORICAL_FEATURES
    print(f"\nFinal feature set: {len(feature_cols)} features "
          f"({len(kept_numerical)} numerical + {len(CATEGORICAL_FEATURES)} categorical)")

    X_train = train_df[feature_cols].values
    X_val   = val_df[feature_cols].values
    X_test  = test_df[feature_cols].values
    y_train = train_df[TARGET].values
    y_val   = val_df[TARGET].values
    y_test  = test_df[TARGET].values

    # Scale numerical features (fit on train only)
    numerical_idx = list(range(len(kept_numerical)))
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

    joblib.dump(scaler,       os.path.join(OUT_DIR, "scaler.joblib"))
    joblib.dump(encoders,     os.path.join(OUT_DIR, "label_encoders.joblib"))
    joblib.dump(feature_cols, os.path.join(OUT_DIR, "feature_cols.joblib"))

    print(f"\nSaved arrays, scaler, encoders, and feature list to {OUT_DIR}")
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Base.csv (BAF NeurIPS 2022)")
    parser.add_argument(
        "--smote", action="store_true",
        help="Apply SMOTE oversampling to the training fold only"
    )
    parser.add_argument(
        "--no-select", action="store_true",
        help="Skip feature selection and use all features"
    )
    args = parser.parse_args()
    load_and_split(smote=args.smote, feature_selection=not args.no_select)
