import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import numpy as np

# Assuming the merged dataframe df_combined is already created as in your previous steps
file_arterial = 'SleepStagingArterialFeatures.xlsx'
file_cardio = 'SleepStagingCardioRespiratoryFeatures.xlsx'
file_statistical = 'SleepStagingStatisticalFeatures.xlsx'

# Load data from each sheet in each file
dfs_arterial = pd.concat(pd.read_excel(file_arterial, sheet_name=None), ignore_index=True)
dfs_cardio = pd.concat(pd.read_excel(file_cardio, sheet_name=None), ignore_index=True)
dfs_statistical = pd.concat(pd.read_excel(file_statistical, sheet_name=None), ignore_index=True)

# Merge all dataframes on 'SubNo' and 'SegNo'
df_combined = pd.merge(dfs_arterial, dfs_cardio, on=['SubNo', 'SegNo'], suffixes=('_arterial', '_cardio'))
df_combined = pd.merge(df_combined, dfs_statistical, on=['SubNo', 'SegNo'], suffixes=('', '_statistical'))

# Handle missing values if any (already handled in previous steps, but let's confirm)
df_combined.fillna(df_combined.mean(), inplace=True)


# Split the dataset into features and labels
X = df_combined.drop(['Class'], axis=1)
y = df_combined['Class']

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Initialize the Random Forest Classifier with the best parameters found earlier
rf = RandomForestClassifier(
    bootstrap=True, max_depth=30, max_features=None,
    min_samples_leaf=4, min_samples_split=10, n_estimators=227, random_state=42
)

# Train the model
rf.fit(X_train, y_train)

# Cross-validation to validate model performance
cv_strategy = StratifiedKFold(n_splits=5)
cross_val_scores = cross_val_score(rf, X_train, y_train, cv=cv_strategy, scoring='accuracy')
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean Cross-Validation Score: {cross_val_scores.mean()}")

# Validation Set Performance
y_val_pred = rf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_report = classification_report(y_val, y_val_pred)
val_roc_auc = roc_auc_score(y_val, rf.predict_proba(X_val), multi_class='ovr')

print(f"Validation Set Accuracy: {val_accuracy}")
print(f"Validation Set Classification Report:\n{val_report}")
print(f"Validation Set ROC AUC Score: {val_roc_auc}")

# Testing Set Performance (Final Evaluation)
y_test_pred = rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_report = classification_report(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test), multi_class='ovr')

print(f"Test Set Accuracy: {test_accuracy}")
print(f"Test Set Classification Report:\n{test_report}")
print(f"Test Set ROC AUC Score: {test_roc_auc}")

# Compare Training and Validation Performance
y_train_pred = rf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_report = classification_report(y_train, y_train_pred)

print(f"Training Set Accuracy: {train_accuracy}")
print(f"Training Set Classification Report:\n{train_report}")

# Check for overfitting by comparing train vs validation performance
accuracy_gap = train_accuracy - val_accuracy
print(f"Train vs Validation Accuracy Gap: {accuracy_gap}")

# Check for Data Leakage
# Ensure no overlap in the data used for training, validation, and testing
'''
# Feature Importance (as in the previous code)
importances = rf.feature_importances_
feature_names = X.columns
important_features = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("Feature Importances:\n", important_features)
'''