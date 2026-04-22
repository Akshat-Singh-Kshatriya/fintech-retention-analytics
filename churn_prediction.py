import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score



df = pd.read_csv(data)


df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

df['LTV_Proxy'] = df['EstimatedSalary'] * (df['Tenure'] / 10)

# One-Hot Encoding for categorical variables
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# Define Features (X) and Target (y)
X = df.drop('Exited', axis=1)
y = df['Exited']

# We still do one initial split to hold out a pure "Test" set for the final business strategy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features (Crucial for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the models
log_reg = LogisticRegression(class_weight='balanced', random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=3.9, random_state=42)

# We want to track multiple metrics during Cross-Validation
scoring_metrics = ['roc_auc', 'recall', 'precision']

cv_results_lr = cross_validate(log_reg, X_train_scaled, y_train, cv=cv_strategy, scoring=scoring_metrics)
print(f"Mean ROC-AUC:  {cv_results_lr['test_roc_auc'].mean():.4f} (+/- {cv_results_lr['test_roc_auc'].std():.4f})")
print(f"Mean Recall:   {cv_results_lr['test_recall'].mean():.4f}")

cv_results_xgb = cross_validate(xgb_model, X_train_scaled, y_train, cv=cv_strategy, scoring=scoring_metrics)
print(f"Mean ROC-AUC:  {cv_results_xgb['test_roc_auc'].mean():.4f} (+/- {cv_results_xgb['test_roc_auc'].std():.4f})")
print(f"Mean Recall:   {cv_results_xgb['test_recall'].mean():.4f}")

# Train the final XGBoost model on the entire training set for business application
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]


print("Final Test ROC-AUC Score:", roc_auc_score(y_test, y_prob_xgb))
print("\nClassification Report (Test Set):\n", classification_report(y_test, y_pred_xgb))


results = X_test.copy()
results['Actual_Churn'] = y_test
results['Churn_Probability'] = y_prob_xgb
results['Predicted_Churn'] = y_pred_xgb

# Define LTV thresholds
high_ltv_threshold = results['LTV_Proxy'].quantile(0.66)
low_ltv_threshold = results['LTV_Proxy'].quantile(0.33)

def assign_nbo_strategy(row):
    if row['Predicted_Churn'] == 1:
        if row['LTV_Proxy'] >= high_ltv_threshold:
            return "Premium Retention: Waive Fees & Assign RM"
        elif row['LTV_Proxy'] >= low_ltv_threshold:
            return "Standard Retention: 5% Cashback Offer"
        else:
            return "Low Value: No Action (Let Churn)"
    else:
        return "Safe: Cross-sell/Up-sell Opportunities"

results['NBO_Strategy'] = results.apply(assign_nbo_strategy, axis=1)

at_risk_customers = results[results['Predicted_Churn'] == 1]
print("\nSample Strategy for At-Risk Customers:")
print(at_risk_customers[['LTV_Proxy', 'Churn_Probability', 'NBO_Strategy']].head(10).to_string())
