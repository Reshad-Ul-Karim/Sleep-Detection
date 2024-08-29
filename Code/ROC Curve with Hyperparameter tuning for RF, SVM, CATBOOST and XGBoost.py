import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo_main.csv")
drop_columns = ['SubNo', "SegNo", "Class", 'averageTeagerEnergy', 'harmonicMean', 'svdPPI',
                'averageTeagerEnergy_statistical', 'harmonicMean_statistical', 'svdPPG']
X = df.drop(drop_columns, axis=1)
y = df["Class"]

# Binarize the output for ROC curves
y_bin = label_binarize(y, classes=np.unique(y))
n_classes = y_bin.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=150, stratify=y)

# Function to plot ROC curve
def plot_roc_curve(y_test, y_prob, model_name, ax):
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"{model_name} Class {i}").plot(ax=ax, alpha=0.3)

# Open a file to save all results
with open("All_Model_Results.txt", "w") as results_file:

    # Random Forest
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
    y_pred_rf = best_rf.predict(X_test)
    y_prob_rf = best_rf.predict_proba(X_test)

    # Cross-Validation Scores
    cv_scores_rf = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='accuracy')

    # Print and save results
    results_file.write("Random Forest Results:\n")
    results_file.write(f'Accuracy: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%\n')
    results_file.write(f'Cross-Validation Scores: {cv_scores_rf}\n')
    results_file.write(f'Average CV Score: {np.mean(cv_scores_rf):.4f}\n')
    results_file.write(classification_report(y_test, y_pred_rf, zero_division=1))
    results_file.write("\n" + "="*80 + "\n")

    # SVM
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
            probability=True,  # Enable probability estimates for ROC curve
            random_state=160
        ))
    ])

    best_svm_model.fit(X_train, y_train)
    y_pred_svm = best_svm_model.predict(X_test)
    y_prob_svm = best_svm_model.predict_proba(X_test)

    # Cross-Validation Scores
    cv_scores_svm = cross_val_score(best_svm_model, X_train, y_train, cv=5, scoring='accuracy')

    # Print and save results
    results_file.write("SVM Results:\n")
    results_file.write(f'Accuracy: {accuracy_score(y_test, y_pred_svm) * 100:.2f}%\n')
    results_file.write(f'Cross-Validation Scores: {cv_scores_svm}\n')
    results_file.write(f'Average CV Score: {np.mean(cv_scores_svm):.4f}\n')
    results_file.write(classification_report(y_test, y_pred_svm, zero_division=1))
    results_file.write("\n" + "="*80 + "\n")

    # XGBoost
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
    y_pred_xgb = best_xgb.predict(X_test)
    y_prob_xgb = best_xgb.predict_proba(X_test)

    # Cross-Validation Scores
    cv_scores_xgb = cross_val_score(best_xgb, X_train, y_train, cv=5, scoring='accuracy')

    # Print and save results
    results_file.write("XGBoost Results:\n")
    results_file.write(f'Accuracy: {accuracy_score(y_test, y_pred_xgb) * 100:.2f}%\n')
    results_file.write(f'Cross-Validation Scores: {cv_scores_xgb}\n')
    results_file.write(f'Average CV Score: {np.mean(cv_scores_xgb):.4f}\n')
    results_file.write(classification_report(y_test, y_pred_xgb, zero_division=1))
    results_file.write("\n" + "="*80 + "\n")

    # CatBoost
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
    y_pred_catboost = best_catboost.predict(X_test)
    y_prob_catboost = best_catboost.predict_proba(X_test)

    # Cross-Validation Scores
    cv_scores_catboost = cross_val_score(best_catboost, X_train, y_train, cv=5, scoring='accuracy')

    # Print and save results
    results_file.write("CatBoost Results:\n")
    results_file.write(f'Accuracy: {accuracy_score(y_test, y_pred_catboost) * 100:.2f}%\n')
    results_file.write(f'Cross-Validation Scores: {cv_scores_catboost}\n')
    results_file.write(f'Average CV Score: {np.mean(cv_scores_catboost):.4f}\n')
    results_file.write(classification_report(y_test, y_pred_catboost, zero_division=1))
    results_file.write("\n" + "="*80 + "\n")

# Plot all ROC curves in a single plot
fig, ax = plt.subplots(figsize=(12, 10))
plot_roc_curve(y_test, y_prob_rf, "Random Forest", ax)
plot_roc_curve(y_test, y_prob_svm, "SVM", ax)
plot_roc_curve(y_test, y_prob_xgb, "XGBoost", ax)
plot_roc_curve(y_test, y_prob_catboost, "CatBoost", ax)

# Plot settings
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves for All Models')
ax.legend(loc="lower right")
plt.savefig('All_Models_ROC_Curve.png')
plt.show()
