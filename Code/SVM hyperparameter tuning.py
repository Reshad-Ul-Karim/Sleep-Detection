import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

df = pd.read_csv("Sleep_Stage_Combo_main.csv")

drop_columns = ['SubNo', "Class", "SegNo"]
X = df.drop(drop_columns, axis=1)
y = df["Class"]

# Create a pipeline with scaling and SVM
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=42))
])

# Define the hyperparameter space

param_distributions = {
    'svm__C': np.logspace(-3, 3, 100),  # Reduced from 1000 to 100
    'svm__kernel': ['rbf', 'poly', 'sigmoid'],
    'svm__degree': [2, 3, 4, 5],
    'svm__gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 100)),  # Reduced from 1000 to 100
    'svm__coef0': np.linspace(-1, 1, 20),  # Reduced from 100 to 20
    'svm__shrinking': [True, False],
    'svm__probability': [False],
    'svm__tol': np.logspace(-4, -1, 100),  # Reduced from 1000 to 100
    'svm__class_weight': ['balanced', None],
    'svm__decision_function_shape': ['ovo'],
    'svm__break_ties': [False]
}

# ... rest of the code remains the same ...

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    svm_pipeline,
    param_distributions=param_distributions,
    n_iter=200,  # Increased number of iterations
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    # Use StratifiedKFold for balanced class representation
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
    random_state=42,
    verbose=2
)

# Fit the random search
random_search.fit(X, y)

# Print the best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score (accuracy):", random_search.best_score_)

# Get the best model
best_model = random_search.best_estimator_

# You can now use best_model for predictions on new data
