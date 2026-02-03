
# Credit Card Fraud Detection using Machine Learning

## Project Overview
This project aims to identify fraudulent credit card transactions within a highly imbalanced dataset. Since fraud accounts for only **0.17%** of the data, we utilize **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the classes and improve model performance.

## Dataset Information
The dataset consists of credit card transactions made by European cardholders in September 2013.
* **Total Rows**: 284,807
* **Total Columns**: 31 (Time, Amount, Class, and PCA-transformed features V1-V28)
* **Target Variable**: `Class` (1 for fraud, 0 for legitimate)

## Methodology
1.  **Preprocessing**: Removed duplicates and dropped the 'Time' column as it was non-essential for this specific analysis.
2.  **Visualization**: Analyzed the distribution of transaction amounts and class imbalance using `ggplot` styled graphs.
3.  **Balancing**: Applied SMOTE to oversample the minority fraud class.
4.  **Modeling**: Evaluated Logistic Regression and **XGBoost** models.

## Final Results (XGBoost)
The XGBoost model achieved high performance in detecting fraud:
* **Accuracy**: ~99.91%
* **Precision**: ~97.28%
* **Recall**: ~90.73%
