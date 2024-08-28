#Based on approach 1, optimise the XGboost 

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from catboost import CatBoostClassifier

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
    'iterations': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'depth': [4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [32, 50, 100, 200],
    'bagging_temperature': [0, 0.5, 1, 2, 3, 5, 7, 10],
    'random_strength': [1, 2, 5, 10, 20],
    'scale_pos_weight': [1, 2, 3, 5]
}

# Initialize the CatBoostClassifier
cb = CatBoostClassifier(loss_function='MultiClass', eval_metric='Accuracy', random_state=150, verbose=0)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=cb,
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
print("Best Parameters for CatBoost:", random_search.best_params_)

# Evaluate the best model on the test set
best_cb = random_search.best_estimator_
y_pred = best_cb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy for best CatBoost: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred, zero_division=1))
