import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import randint, uniform
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_csv('Sleep_Stage_Combo4.csv')

drop_columns = ['SubNo', "SegNo", "Class","Class2"]

X = df.drop(drop_columns, axis=1)
y = df["Class2"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', CatBoostClassifier(random_state=42, verbose=0))
])

# Define the hyperparameter space
import numpy as np
from scipy.stats import randint, uniform

# ... (previous code remains the same)

# Define the hyperparameter space
# ... (previous code remains the same)

# Define the hyperparameter space
param_distributions = {
    'classifier__iterations': randint(100, 1000),
    'classifier__learning_rate': uniform(0.01, 0.3),
    'classifier__depth': randint(4, 10),
    'classifier__l2_leaf_reg': uniform(1, 10),
    'classifier__border_count': randint(32, 255),
    'classifier__bagging_temperature': uniform(0, 1),
    'classifier__random_strength': uniform(0, 1),
    'classifier__grow_policy': ['SymmetricTree', 'Depthwise'],
    'classifier__min_data_in_leaf': randint(1, 20),
    'classifier__rsm': uniform(0.1, 0.9),
    'classifier__leaf_estimation_method': ['Newton', 'Gradient'],
    'classifier__boosting_type': ['Ordered', 'Plain'],
    'classifier__bootstrap_type': ['Bernoulli', 'MVS'],
    'classifier__score_function': ['Cosine', 'L2'],
    'classifier__od_type': ['IncToDec', 'Iter'],
    'classifier__od_wait': randint(10, 50),
    'classifier__leaf_estimation_iterations': randint(1, 10),
}

# Generate random class weights
n_classes = len(np.unique(y))
random_weights = np.random.rand(n_classes)
normalized_weights = random_weights / np.sum(random_weights)
class_weights_dict = {i: w for i, w in enumerate(normalized_weights)}

# Add class_weights to param_distributions
param_distributions['classifier__class_weights'] = [class_weights_dict]



# Set up the randomized search
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

# Perform the randomized search
random_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score_)

# Evaluate on the test set
best_model = random_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test set score:", test_score)