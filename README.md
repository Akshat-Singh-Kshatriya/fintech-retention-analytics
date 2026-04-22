# Fintech Customer Retention & LTV Analytics 

## Executive Summary
This project bridges the gap between predictive machine learning and actionable business strategy. I engineered an end-to-end analytics pipeline using financial services data to predict customer churn. Instead of stopping at model accuracy, this pipeline calculates a proxy for **Customer Lifetime Value (LTV)** to segment at-risk users. The final output is a data-driven **Next-Best-Offer (NBO) strategy** designed to optimize targeted marketing spend by deploying premium retention incentives only to high-value clients.

## Data Source
* **Dataset:** [Bank Customer Churn Prediction (Kaggle)](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset).
* **Features:** Credit Score, Geography, Age, Tenure, Balance, Number of Products, Estimated Salary, and Exited (Churn indicator).

## Tech Stack
* **Language:** Python (Pandas, Numpy)
* **Machine Learning:** scikit-learn, Logistic-Regression, XGBoost-Classifier
* **Validation:** Stratified 5-Fold Cross-Validation

## Key Results & Model Performance
To handle the imbalanced nature of churn data, the pipeline leverages an `XGBClassifier` evaluated via Stratified 5-Fold Cross-Validation to ensure robust generalization.

* **ROC-AUC Score:** 0.83 (Strong capability to distinguish churners from loyal customers)
* **Recall:** 62% (Successfully identified the majority of actual churners)
* **Precision:** 55%

**Business Context of Metrics:** A precision of 55% indicates the presence of false positives. From a consulting perspective, deploying expensive retention packages to all flagged users would result in financial loss. Therefore, the pipeline integrates LTV thresholds to mitigate this risk.

## Business Strategy: Next-Best-Offer (NBO)
The model's probabilities are combined with the calculated LTV to output a targeted strategy for every single customer:

1.  **Premium Retention (High LTV + High Churn Risk):** Waive credit card fees and assign a dedicated Relationship Manager.
2.  **Standard Retention (Medium LTV + High Churn Risk):** Deploy automated, low-cost incentives (e.g., 5% cashback on the next 5 transactions).
3.  **Low Value (Low LTV + High Churn Risk):** No action. Let the customer churn, as the cost of retention exceeds their lifetime value.
4.  **Safe (Low Churn Risk):** Target for cross-selling and up-selling opportunities.

*Sample output from the pipeline:*
| LTV_Proxy | Churn_Probability | NBO_Strategy |
| :--- | :--- | :--- |
| 94,626.07 | 0.703 | Premium Retention: Waive Fees & Assign RM |
| 46,771.73 | 0.861 | Standard Retention: 5% Cashback Offer |
| 10,559.77 | 0.976 | Low Value: No Action (Let Churn) |

## How to Clone & Run

**1. Clone the repository:**
```bash
git clone [https://github.com/Akshat-Singh-Kshatriya/fintech-retention-analytics.git](https://github.com/Akshat-Singh-Kshatriya/fintech-retention-analytics.git)
cd fintech-retention-analytics
```
**2. Run the Model**
```bash
python churn_prediction
```
