import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")

# Prepare the data
drop_columns = ['SubNo', "SegNo", "Class", "Class2", 'averageTeagerEnergy', 'harmonicMean', 'svdPPI',
                'averageTeagerEnergy_statistical', 'harmonicMean_statistical', 'svdPPG']
X = df.drop(drop_columns, axis=1)
y = df["Class2"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Create a pipeline with scaling and SVM
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=160))
])

# Define a more refined hyperparameter grid
param_grid = {
    'svm__C': [1, 10, 100, 1000],  # Exploring a wider range
    'svm__gamma': ['scale', 'auto', 0.01, 0.001],  # Adding more specific gamma values
    'svm__kernel': ['rbf', 'poly'],  # Testing RBF and Polynomial kernels
    'svm__degree': [2, 3, 4],  # Exploring polynomial degrees
    'svm__class_weight': ['balanced'],  # Ensuring class balance
    'svm__shrinking': [True, False],  # Testing both shrinking options
}

# Initialize the GridSearchCV object with more refined parameters
grid_search = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation for robustness
    scoring='accuracy',
    n_jobs=-1,  # Utilize all cores
    verbose=2
)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and score
best_params_grid = grid_search.best_params_
best_score_grid = grid_search.best_score_

# Evaluate the best model on the test set
best_model_grid = grid_search.best_estimator_
y_pred_best_grid = best_model_grid.predict(X_test)
final_accuracy_grid = accuracy_score(y_test, y_pred_best_grid)
final_class_report_grid = classification_report(y_test, y_pred_best_grid)

# Print results
print("Best parameters from GridSearchCV:", best_params_grid)
print("Best cross-validation accuracy from GridSearchCV:", best_score_grid)
print("Test set accuracy with best parameters:", final_accuracy_grid)
print("Classification report for test set:\n", final_class_report_grid)
