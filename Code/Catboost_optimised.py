import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform

# Load data
df = pd.read_csv('Sleep_Stage_Combo.csv')
X = df.drop(['SubNo', 'SegNo', 'Class'], axis=1)
y = df['Class']

# Split data with stratified sampling to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Initialize the CatBoost Classifier
model = CatBoostClassifier(verbose=0, auto_class_weights='Balanced')

# Define parameter distribution for Randomized Search
param_distributions = {
    'iterations': randint(100, 800),
    'depth': randint(4, 10),
    'learning_rate': uniform(0.01, 0.2),
    'l2_leaf_reg': randint(1, 10),
    'border_count': randint(50, 255),
    'random_strength': uniform(0.0, 1.0),
    'bagging_temperature': uniform(0.0, 1.0),
    'od_type': ['IncToDec', 'Iter'],
    'od_wait': randint(10, 50)
}

# Setup the RandomizedSearchCV with StratifiedKFold
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(model, param_distributions, n_iter=50, cv=cv_strategy, scoring='accuracy', verbose=2, random_state=42)

# Fit the model
random_search.fit(X_train, y_train)

# Best model evaluation
best_model = random_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
best_params = random_search.best_params_

print(f"Best Model Accuracy: {accuracy}")
print(f"Best Model Parameters: {best_params}")
