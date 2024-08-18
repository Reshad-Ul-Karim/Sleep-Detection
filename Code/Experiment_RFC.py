import pandas as pd
from scipy.stats import randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the datasets from each .xlsx file
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
df_combined.drop(['Class_cardio', 'Class_arterial'], axis=1, inplace=True)
# print all the subNo
print("\n".join([str(i) for i in df_combined['SubNo']]))
# # Handle missing values if any (already handled in previous steps, but let's confirm)
# df_combined.fillna(df_combined.mean(), inplace=True)
#
# Split the dataset into features and labels
X = df_combined.drop(['Class', 'SubNo', 'SegNo'], axis=1)
y = df_combined['Class']
# export the data to csv
df_combined.to_csv('Sleep_Stage_Combo2.csv', index=False)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test), multi_class='ovr')

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"ROC AUC Score: {roc_auc}")

# Optional: Hyperparameter tuning using RandomizedSearchCV
random_param_space = {
    'n_estimators': randint(100, 500),
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

cv_strategy = StratifiedKFold(n_splits=5)

random_search = RandomizedSearchCV(
    rf, random_param_space, n_iter=100, scoring='accuracy', cv=cv_strategy,
    n_jobs=-1, random_state=42, verbose=2
)
random_search.fit(X_train, y_train)

# Best parameters and model evaluation
best_params = random_search.best_params_
print("Best parameters found: ", best_params)

# Grid Search to fine-tune around the best results from Randomized Search
param_grid = {
    'n_estimators': [best_params['n_estimators'] - 50, best_params['n_estimators'], best_params['n_estimators'] + 50],
    'max_depth': [best_params['max_depth']],
    'min_samples_split': [best_params['min_samples_split'] - 1, best_params['min_samples_split'],
                          best_params['min_samples_split'] + 1],
    'min_samples_leaf': [best_params['min_samples_leaf'] - 1, best_params['min_samples_leaf'],
                         best_params['min_samples_leaf'] + 1],
    'max_features': [best_params['max_features']],
    'bootstrap': [best_params['bootstrap']]
}

grid_search = GridSearchCV(
    rf, param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1, verbose=2
)
grid_search.fit(X_train, y_train)

# Final model evaluation
final_model = grid_search.best_estimator_
y_pred_final = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_final)
final_report = classification_report(y_test, y_pred_final)
final_roc_auc = roc_auc_score(y_test, final_model.predict_proba(X_test), multi_class='ovr')

print(f"Final Model Accuracy: {final_accuracy}")
print(f"Final Model Classification Report:\n{final_report}")
print(f"Final ROC AUC Score: {final_roc_auc}")

# Feature Importance
importances = final_model.feature_importances_
feature_names = X.columns
important_features = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("Feature Importances:\n", important_features)
