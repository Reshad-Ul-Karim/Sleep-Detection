import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")

# Define features and target
drop_columns = ['SubNo', "SegNo", "Class", "Class2"]
X = df.drop(drop_columns, axis=1)
y = df["Class2"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=150)

# Redefine the parameter grids for Random Forest and XGBoost
param_grid_rf_advanced = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

param_grid_xgb_advanced = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [6, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'colsample_bytree': [0.3, 0.7, 1.0],
    'subsample': [0.5, 0.7, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_lambda': [1, 1.5, 2.0]
}

# Perform advanced Grid Search for each model
grid_search_rf_advanced = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf_advanced, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search_rf_advanced.fit(X_train, y_train)

grid_search_xgb_advanced = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False), param_grid_xgb_advanced, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search_xgb_advanced.fit(X_train, y_train)

# Evaluate the best models
best_rf_advanced = grid_search_rf_advanced.best_estimator_
y_pred_rf_advanced = best_rf_advanced.predict(X_test)
accuracy_rf_advanced = accuracy_score(y_test, y_pred_rf_advanced)
classification_report_rf_advanced = classification_report(y_test, y_pred_rf_advanced, zero_division=1)

best_xgb_advanced = grid_search_xgb_advanced.best_estimator_
y_pred_xgb_advanced = best_xgb_advanced.predict(X_test)
accuracy_xgb_advanced = accuracy_score(y_test, y_pred_xgb_advanced)
classification_report_xgb_advanced = classification_report(y_test, y_pred_xgb_advanced, zero_division=1)

# Compile advanced tuning results
advanced_results = {
    "Random Forest Advanced": {"accuracy": accuracy_rf_advanced, "classification_report": classification_report_rf_advanced},
    "XGBoost Advanced": {"accuracy": accuracy_xgb_advanced, "classification_report": classification_report_xgb_advanced}
}

print(advanced_results)
