import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('Sleep_Stage_Combo.csv')

# Preprocessing
drop_columns = ['SubNo', 'SegNo', 'Class']
X = df.drop(drop_columns, axis=1)
y = df["Class"]

# Handling missing values if any
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Standardizing features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter grid
random_grid = {
    # Updated to a realistic example setup
    'n_estimators': [100, 300, 500, 800, 1200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'colsample_bytree': [0.3, 0.7, 1.0],
    'subsample': [0.5, 0.75, 1.0],
    'gamma': [0.0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}

# Initialize the XGBClassifier
xgb = XGBClassifier(eval_metric='mlogloss', enable_categorical=True)


# Perform Randomized Search with StratifiedKFold
cv_strategy = StratifiedKFold(n_splits=5)
random_search = RandomizedSearchCV(estimator=xgb, param_distributions=random_grid, n_iter=100, cv=cv_strategy, verbose=2,
                                   random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Best parameters and model evaluation
print("Best parameters found: ", random_search.best_params_)
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred, multi_class='ovr'))

