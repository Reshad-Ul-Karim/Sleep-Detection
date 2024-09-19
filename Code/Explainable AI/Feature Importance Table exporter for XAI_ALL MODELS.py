import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Load the data
df = pd.read_csv('NSys/Sleep_Stage_Combo2.csv')

# Drop unnecessary columns
drop_columns = ['SubNo', "SegNo", "Class", "Class2", 'averageTeagerEnergy', 'harmonicMean', 'svdPPI',
                'averageTeagerEnergy_statistical', 'harmonicMean_statistical', 'svdPPG']
X = df.drop(drop_columns, axis=1)
y = df["Class2"]

# Hyperparameters for models
hyperparameters_CB = {
    'bagging_temperature': 0.4175,
    'boosting_type': 'Ordered',
    'bootstrap_type': 'MVS',
    'border_count': 65,
    'class_weights': {0: 0.155, 1: 0.319, 2: 0.15, 3: 0.376},
    'depth': 9,
    'iterations': 858,
    'l2_leaf_reg': 2.85,
    'learning_rate': 0.289,
    'min_data_in_leaf': 7,
    'od_type': 'Iter',
    'od_wait': 38,
    'random_strength': 0.767,
    'rsm': 0.462
}

hyperparameters_XGB = {
    "objective": 'multi:softmax',
    "eval_metric": 'mlogloss',
    "random_state": 150,
    "subsample": 0.9,
    "reg_lambda": 0.04,
    "reg_alpha": 0.7,
    "n_estimators": 600,
    "max_depth": 9,
    "learning_rate": 0.05,
    "colsample_bytree": 0.9
}

hyperparameters_RF = {
    "n_estimators": 300,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "max_features": 'log2',
    "max_depth": 20,
    "criterion": 'entropy',
    "bootstrap": False,
    "random_state": 150
}

# Models dictionary
models = {
    "RandomForest": RandomForestClassifier(**hyperparameters_RF),
    "CatBoost": CatBoostClassifier(**hyperparameters_CB),
    "XGBoost": XGBClassifier(**hyperparameters_XGB),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            tol=0.12,
            shrinking=False,
            kernel='rbf',
            gamma='scale',
            degree=3,
            coef0=0.5,
            class_weight='balanced',
            C=100.0,
            probability=True,  # Enable probability estimates for ROC curve
            random_state=160
        ))
    ])
}

# Initialize dictionary to store feature importances
all_importances = {model: {feature: 0 for feature in X.columns} for model in models}

# Train models and gather feature importances
for name, model in models.items():
    for i in range(5):  # Run the process 5 times to average feature importance
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=42 * i)
        model.fit(X_train, y_train)

        if name == "SVM":
            # For SVM, use permutation importance
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=50 * i)
            importances = dict(zip(X.columns, perm_importance.importances_mean))  # Convert to dictionary
        else:
            # For other models, use feature importances directly
            importances = pd.Series(model.feature_importances_, index=X.columns).to_dict()

        # Adjust CatBoost feature importances (if required)
        if name == "CatBoost":
            for feature in importances:
                importances[feature] /= 100

        # Accumulate feature importances
        for feature in importances:
            all_importances[name][feature] += importances[feature]

# Convert all_importances to DataFrame
importances_df = pd.DataFrame(all_importances)

# Transpose the DataFrame to get models as columns and features as rows
importances_df = importances_df.transpose()

# Save the DataFrame to CSV
importances_df.to_csv('model_importances.csv')
