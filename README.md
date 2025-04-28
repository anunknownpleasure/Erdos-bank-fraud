# Bank Account Fraud Detection - Erdős Bootcamp  

## 👥 Our Team  
We are a team — Arpith Shanbhag and Maksim Kosmakov — participating in the  in the **Erdős Bootcamp**, working on the **Bank Account Fraud (BAF) Kaggle competition**. Our goal is to develop fair, explainable, and effective ML models for fraud detection, tackling challenges such as **class imbalance, bias, and temporal shifts**.  

## 🚀 Project Overview  
This repository contains our work for the **Bank Account Fraud (BAF) Kaggle competition**, which is based on the **NeurIPS 2022 BAF dataset**. The competition provides a **realistic, privacy-preserving, and imbalanced** fraud detection dataset.  

## 📌 Dataset  
The **BAF dataset** consists of **six synthetic tabular datasets** designed to simulate real-world fraud detection scenarios. It features:  
- **Extremely imbalanced data** with a low prevalence of fraud cases.  
- **Controlled biases** across different datasets.  
- **Temporal aspects** with observed distribution shifts.  
- **Privacy-preserving features**, including **differential privacy techniques** and a **CTGAN-generated dataset**.  

For more details, check out the [official Kaggle competition page](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022?select=Base.csv).  

## 📊 Objective  
The primary objective is to build **fair, explainable, and effective ML models** for fraud detection. The key challenges include handling extreme **class imbalance**.  


## 📈 Evaluation Metric  
Given the dataset's extreme class imbalance, traditional accuracy is not meaningful. Instead, the competition recommends: 
- **F1-Score**
- **Area Under the Precision-Recall Curve (AUPRC)**  
- **Recall** 
- **Precision**



 ## Stakeholders

This project is valuable to multiple stakeholders, including:

- **Financial Institutions & Banks**: To improve fraud detection and reduce financial losses.
- **Credit Card Companies**: To prevent fraudulent transactions and enhance security.
- **Consumers**: To protect customers from unauthorized transactions and fraudulent charges.
- **Data Scientists & Machine Learning Engineers**: To develop and refine models for fraud detection.
- **Regulatory Agencies**: To ensure compliance with financial fraud prevention standards.

 ## 🏗️Repository Structure
- EDA/: Contains exploratory data analysis notebooks and reports.
- Models/: Contains model selection experiments and results.

## Approach:

Our primary focus was to maximize the F1 score. To address the highly imbalanced dataset, we applied:
- Undersampling techniques
- Oversampling techniques using SMOTE

We considered several models, including:
- Logistic Regression
- Decision Tree
- K-Neighbors Classifier

The best F1 score was achieved using Logistic Regression with a threshold of 0.89, resulting in an F1 score of 0.21.

Further details, including precision-recall curves and clustering analysis, can be found in the model selection notebook (Models/model_selection.ipynb).


This project is licensed under the MIT License.
