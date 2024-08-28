#Approach 1 based XGBoost hyperparameter

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")

# Split the dataset into training and testing sets
drop_columns = ['SubNo', "SegNo", "Class", "Class2", 'averageTeagerEnergy', 'harmonicMean', 'svdPPI',
                'averageTeagerEnergy_statistical', 'harmonicMean_statistical', 'svdPPG']
X = df.drop(drop_columns, axis=1)
y = df["Class2"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=150)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'reg_alpha': [0, 0.01, 0.05, 0.1, 1],
    'reg_lambda': [0.01, 0.05, 0.1, 0.5, 1]
}

# Initialize the XGBClassifier
xgb = XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', random_state=150)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=100,  # Number of parameter settings that are sampled
    cv=5,  # 5-fold cross-validation
    verbose=2,
    random_state=150,
    n_jobs=-1  # Use all available cores
)

# Fit the RandomizedSearchCV to find the best parameters
random_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters for XGBoost:", random_search.best_params_)

# Evaluate the best model on the test set
best_xgb = random_search.best_estimator_
y_pred = best_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy for best XGBoost: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred, zero_division=1))

with open("XGBoost_model_evaluation.txt", "w") as file:
    file.write(f'Accuracy for best XGBoost: {accuracy * 100:.2f}%\n')
    file.write(classification_report(y_test, y_pred, zero_division=1))
