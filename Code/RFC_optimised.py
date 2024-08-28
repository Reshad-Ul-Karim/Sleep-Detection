import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import numpy as np

# Load and prepare data
df = pd.read_csv('Sleep_Stage_Combo.csv')
X = df.drop(['SubNo', 'SegNo', 'Class'], axis=1)
y = df['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the Random Forest Classifier with the best parameters from RandomizedSearchCV
best_rf_params = {
    'bootstrap': True,
    'max_depth': 30,
    'max_features': None,
    'min_samples_leaf': 4,
    'min_samples_split': 10,
    'n_estimators': 227
}

rf = RandomForestClassifier(**best_rf_params, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test), multi_class='ovr')

print(f"Final Model Accuracy: {accuracy}")
print(f"Final Model Classification Report:\n{report}")
print(f"Final ROC AUC Score: {roc_auc}")

# Optional: Grid Search to fine-tune further around the best parameters if needed
cv_strategy = StratifiedKFold(n_splits=5)

grid_param_space = {
    'n_estimators': [best_rf_params['n_estimators'] - 50, best_rf_params['n_estimators'], best_rf_params['n_estimators'] + 50],
    'max_depth': [best_rf_params['max_depth']],
    'min_samples_split': [9, 10, 11],  # Adjusting based on the found best value
    'min_samples_leaf': [3, 4, 5],  # Adjusting based on the found best value
    'max_features': [best_rf_params['max_features']],
    'bootstrap': [best_rf_params['bootstrap']]
}

grid_search = GridSearchCV(
    rf, grid_param_space, cv=cv_strategy, scoring='balanced_accuracy', n_jobs=-1, verbose=2, error_score='raise'
)
grid_result = grid_search.fit(X_train, y_train)

# Final model evaluation after Grid Search
best_model = grid_result.best_estimator_
y_pred_final = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_final)
final_report = classification_report(y_test, y_pred_final)
final_roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr')

print(f"Grid Search Final Model Accuracy: {final_accuracy}")
print(f"Grid Search Final Model Classification Report:\n{final_report}")
print(f"Grid Search Final ROC AUC Score: {final_roc_auc}")

# Feature Importance
importances = best_model.feature_importances_
feature_names = X.columns
important_features = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("Feature Importances:\n", important_features)
