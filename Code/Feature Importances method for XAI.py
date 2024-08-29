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
df = pd.read_csv("Sleep_Stage_Combo_main.csv")
drop_columns = ['SubNo', "SegNo", "Class", 'averageTeagerEnergy', 'harmonicMean', 'svdPPI',
                'averageTeagerEnergy_statistical', 'harmonicMean_statistical', 'svdPPG']
X = df.drop(drop_columns, axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=150, stratify=y)

# Random Forest Feature Importance
best_rf = RandomForestClassifier(
    n_estimators=600,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=20,
    criterion='entropy',
    bootstrap=False,
    random_state=90
)

best_rf.fit(X_train, y_train)
rf_importances = best_rf.feature_importances_

# Plot Random Forest feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=rf_importances, y=X.columns)
plt.title('Random Forest Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('rf_feature_importance.png')
plt.show()

# SVM Feature Importance (Permutation Importance for non-linear kernel)
best_svm_model = Pipeline([
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

best_svm_model.fit(X_train, y_train)
result = permutation_importance(best_svm_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
svm_importances = result.importances_mean

# Plot SVM feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=svm_importances, y=X.columns)
plt.title('SVM Feature Importances (Permutation)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('svm_feature_importance.png')
plt.show()

# XGBoost Feature Importance
best_xgb = XGBClassifier(
    objective='multi:softmax',
    eval_metric='mlogloss',
    random_state=150,
    subsample=0.9,
    reg_lambda=0.04,
    reg_alpha=0.7,
    n_estimators=600,
    max_depth=9,
    learning_rate=0.05,
    gamma=0,
    colsample_bytree=0.9
)

best_xgb.fit(X_train, y_train)
xgb_importances = best_xgb.feature_importances_

# Plot XGBoost feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=xgb_importances, y=X.columns)
plt.title('XGBoost Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('xgb_feature_importance.png')
plt.show()

# CatBoost Feature Importance
best_params = {
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

best_catboost = CatBoostClassifier(
    **best_params,
    random_state=42,
    verbose=0
)

best_catboost.fit(X_train, y_train)
catboost_importances = best_catboost.get_feature_importance()

# Plot CatBoost feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=catboost_importances, y=X.columns)
plt.title('CatBoost Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('catboost_feature_importance.png')
plt.show()
