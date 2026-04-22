## Context

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

## Content

The dataset contains transactions made by credit cards in September 2013 by European cardholders.

This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'.  

- **Feature 'Time'** contains the seconds elapsed between each transaction and the first transaction in the dataset.  
- **Feature 'Amount'** is the transaction amount, which can be used for example-dependent cost-sensitive learning.  
- **Feature 'Class'** is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Evaluation Metrics

Given the class imbalance ratio, we are measuring the accuracy using the **Area Under the Precision-Recall Curve (AUPRC)**. Confusion matrix accuracy is not meaningful for unbalanced classification.

## Key Performance Indicators (KPIs)

- **AUPRC (Area Under Precision-Recall Curve)**: Measures how well the model identifies fraudulent transactions while minimizing false positives.
- **Recall (Sensitivity)**: The percentage of actual frauds that are correctly identified.
- **Precision**: The percentage of predicted frauds that are actually fraudulent.
- **F1 Score**: The harmonic mean of precision and recall, balancing both metrics.
- **False Positive Rate (FPR)**: The rate at which legitimate transactions are incorrectly flagged as fraud.
- **False Negative Rate (FNR)**: The rate at which fraudulent transactions are missed.

 ## Stakeholders

This project is valuable to multiple stakeholders, including:

- **Financial Institutions & Banks**: To improve fraud detection and reduce financial losses.
- **Credit Card Companies**: To prevent fraudulent transactions and enhance security.
- **Consumers**: To protect customers from unauthorized transactions and fraudulent charges.
- **Data Scientists & Machine Learning Engineers**: To develop and refine models for fraud detection.
- **Regulatory Agencies**: To ensure compliance with financial fraud prevention standards.


This project is licensed under the MIT License.

---

## Setup

```bash
git clone <repo-url>
cd Erdos-bank-fraud

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle:
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in DATA/
```

## Usage

### 1. Explore the data

```bash
jupyter notebook EDA/fraud_eda.ipynb
```

### 2. Preprocess

```bash
# Default: no oversampling (models use class_weight='balanced' / scale_pos_weight)
python DATA/preprocess.py

# Optional: apply SMOTE oversampling to the training fold
python DATA/preprocess.py --smote
```

### 3. Train models

```bash
python MODELS/train_models.py
```

Trains four models — Logistic Regression, Random Forest, LightGBM, XGBoost — using 5-fold stratified cross-validation and prints a comparison table.

### 4. Evaluate

```bash
python MODELS/evaluate.py
```

Reports all KPIs on the held-out test set and saves precision-recall and ROC curves to `outputs/figures/`.

## Results

| Model               | CV AUPRC (mean ± std) | Val AUPRC |
|---------------------|-----------------------|-----------|
| Logistic Regression | TBD                   | TBD       |
| Random Forest       | TBD                   | TBD       |
| LightGBM            | TBD                   | TBD       |
| XGBoost             | TBD                   | TBD       |

_Run the pipeline with your copy of `creditcard.csv` to populate this table._

## Reproducibility

All random states are fixed to `RANDOM_STATE = 42`. Results are fully deterministic given the same `creditcard.csv` input. The fitted `StandardScaler` is saved alongside the processed arrays in `outputs/processed/scaler.joblib` for consistent inference on new data.
