import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from scipy.stats import randint, uniform
import numpy as np

# Load and prepare data
df = pd.read_csv('Sleep_Stage_Combo.csv')
X = df.drop(['SubNo', 'SegNo', 'Class'], axis=1)
y = df['Class']

# Initialize classifier
rf = RandomForestClassifier()

# Randomized Search to explore a broad range
random_param_space = {
    'n_estimators': randint(100, 500),  # More focused range
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Using StratifiedKFold for better handling of imbalanced classes
cv_strategy = StratifiedKFold(n_splits=5)

random_search = RandomizedSearchCV(
    rf, random_param_space, n_iter=100, scoring='balanced_accuracy', cv=cv_strategy,
    n_jobs=-1, random_state=42, verbose=2
)
random_result = random_search.fit(X, y)
print('Random Search Best Parameters:', random_result.best_params_)
print('Random Search Best Score:', random_result.best_score_)

# Grid Search to fine-tune around the best results from Randomized Search
grid_param_space = {
    'n_estimators': [random_result.best_params_['n_estimators'] - 50, random_result.best_params_['n_estimators'], random_result.best_params_['n_estimators'] + 50],
    'max_depth': [random_result.best_params_['max_depth']],
    'min_samples_split': [random_result.best_params_['min_samples_split'] - 1, random_result.best_params_['min_samples_split'], random_result.best_params_['min_samples_split'] + 1],
    'min_samples_leaf': [random_result.best_params_['min_samples_leaf'] - 1, random_result.best_params_['min_samples_leaf'], random_result.best_params_['min_samples_leaf'] + 1],
    'max_features': [random_result.best_params_['max_features']],
    'bootstrap': [random_result.best_params_['bootstrap']]
}

grid_search = GridSearchCV(
    rf, grid_param_space, cv=cv_strategy, scoring='balanced_accuracy', n_jobs=-1, verbose=2
)
grid_result = grid_search.fit(X, y)
print('Grid Search Best Parameters:', grid_result.best_params_)
print('Grid Search Best Score:', grid_result.best_score_)
