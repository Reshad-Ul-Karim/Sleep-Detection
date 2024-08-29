import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")
drop_columns = ['SubNo', "SegNo", "Class", 'averageTeagerEnergy', 'harmonicMean', 'svdPPI',
                'averageTeagerEnergy_statistical', 'harmonicMean_statistical', 'svdPPG', "Class2"]
X = df.drop(drop_columns, axis=1)
y = df["Class2"]

hyperparameters_CB = {
    'bagging_temperature': 0.41750608402789213,
    'boosting_type': 'Ordered',
    'bootstrap_type': 'MVS',
    'border_count': 65,
    'class_weights': {0: 0.15504341779626502, 1: 0.31884044106753684, 2: 0.1498282622340735, 3: 0.3762878789021247},
    'depth': 9,
    'grow_policy': 'SymmetricTree',
    'iterations': 858,
    'l2_leaf_reg': 2.8547290957672247,
    'leaf_estimation_iterations': 2,
    'leaf_estimation_method': 'Gradient',
    'learning_rate': 0.28977697282757586,
    'min_data_in_leaf': 7,
    'od_type': 'Iter',
    'od_wait': 38,
    'random_strength': 0.7671882889311269,
    'rsm': 0.46173782224831794,
    'score_function': 'Cosine'
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
    "gamma": 0,
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
cb = CatBoostClassifier(**hyperparameters_CB)
xgb = XGBClassifier(**hyperparameters_XGB)
rf = RandomForestClassifier(**hyperparameters_RF)
svm = Pipeline([
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
        probability=True,
        random_state=160
    ))
])
models = [
    ("XGBoost", xgb, "Blues"),
    ("Random Forest", rf, "Purples"),
    ("CatBoost", cb, "Reds"),
    ("SVM", svm, "Oranges")
]

# Initialize dictionaries to store cumulative feature importance for each model
feature_importance_sums = {name: {feature: 0 for feature in X.columns} for name, _, _ in models}

for i in range(10):
    for name, model, color in models:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=50 * i, stratify=y)
        model.fit(X_train, y_train)

        # Get feature importance
        if name == "SVM":
            # For SVM, use permutation importance
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=50 * i)
            feature_importance = perm_importance.importances_mean
            feature_importance = zip(X.columns, feature_importance)
        else:
            feature_importance = model.feature_importances_
            feature_importance = zip(X.columns, feature_importance)
        for feature, importance in feature_importance:
            # Add to cumulative sum
            feature_importance_sums[name][feature] += importance

        # Add to cumulative sum

# Calculate average feature importance
feature_importance_avg = {name: {feature: importance / 10 for feature, importance in importance_dict.items()}
                          for name, importance_dict in feature_importance_sums.items()}

# sort the feature_importance_avg
for name, importance_dict in feature_importance_avg.items():
    feature_importance_avg[name] = {k: v for k, v in
                                    sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
# Create a DataFrame with feature names and importance for each model
feature_importance_df = pd.DataFrame(feature_importance_avg)
# export the feature_importance_df with model name as column and feature name as content
feature_importance_df.to_csv("feature_importance_df.csv")