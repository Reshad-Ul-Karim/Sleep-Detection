import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import defaultdict
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
# Assuming you have your data in a DataFrame `df` and the target variable in `target`
df = pd.read_csv('Sleep_Stage_Combo2.csv')
drop_columns = ['SubNo', "SegNo", "Class", "Class2", 'averageTeagerEnergy', 'harmonicMean', 'svdPPI',
                'averageTeagerEnergy_statistical', 'harmonicMean_statistical', 'svdPPG']
# train test split
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

all_importances = {model: {feature: 0 for feature in X.columns} for model in models}

for name, model in models.items():
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=42 * i)
        model.fit(X_train, y_train)
        if name == "SVM":
            # For SVM, use permutation importance
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=50 * i)
            importances = perm_importance.importances_mean
            importances = zip(X.columns, importances)
        else:
            importances = pd.Series(model.feature_importances_, index=X.columns)
        importances_by_index = importances.to_dict()
        for feature in importances_by_index:
            if name == "CatBoost":
                importances_by_index[feature] /= 100
            all_importances[name][feature] += importances_by_index[feature]

# Convert all_importances to DataFrame
importances_df = pd.DataFrame(all_importances)

# Transpose the DataFrame to get models as columns and regions as rows
importances_df = importances_df.transpose()

# Save the DataFrame to CSV
importances_df.to_csv('model_importances.csv')