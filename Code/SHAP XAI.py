import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import shap

# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")
drop_columns = ['SubNo', "SegNo", "Class", 'averageTeagerEnergy', 'harmonicMean', 'svdPPI',
                'averageTeagerEnergy_statistical', 'harmonicMean_statistical', 'svdPPG']
X = df.drop(drop_columns, axis=1)
y = df["Class"]

# Binarize the output for ROC curves
y_bin = label_binarize(y, classes=np.unique(y))
n_classes = y_bin.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=150, stratify=y)

# SHAP for Random Forest
best_rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features='log2',
    max_depth=20,
    criterion='entropy',
    bootstrap=False,
    random_state=150
)

best_rf.fit(X_train, y_train)
explainer_rf = shap.TreeExplainer(best_rf)
shap_values_rf = explainer_rf.shap_values(X_test)

# SHAP Summary Plot for Random Forest
shap.summary_plot(shap_values_rf, X_test, feature_names=X.columns)
plt.savefig('shap_summary_rf.png')

# SHAP for SVM with reduced background samples
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
        probability=True,  # Enable probability estimates for SHAP
        random_state=160
    ))
])

best_svm_model.fit(X_train, y_train)

# Reduce background data size for SHAP
background = shap.sample(X_train, 100)  # Summarize background as 100 samples

explainer_svm = shap.KernelExplainer(lambda x: best_svm_model.predict_proba(x), background)
shap_values_svm = explainer_svm.shap_values(X_test)

# SHAP Summary Plot for SVM
shap.summary_plot(shap_values_svm, X_test, feature_names=X.columns)
plt.savefig('shap_summary_svm.png')

# SHAP for XGBoost
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
explainer_xgb = shap.TreeExplainer(best_xgb)
shap_values_xgb = explainer_xgb.shap_values(X_test)

# SHAP Summary Plot for XGBoost
shap.summary_plot(shap_values_xgb, X_test, feature_names=X.columns)
plt.savefig('shap_summary_xgb.png')

# SHAP for CatBoost
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
explainer_catboost = shap.TreeExplainer(best_catboost)
shap_values_catboost = explainer_catboost.shap_values(X_test)

# SHAP Summary Plot for CatBoost
shap.summary_plot(shap_values_catboost, X_test, feature_names=X.columns)
plt.savefig('shap_summary_catboost.png')
