import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

from itertools import cycle

# Load dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")
y = df["Class2"]

# Binarize the output for ROC curve generation
y_bin = label_binarize(y, classes=[0, 1, 2, 3])
n_classes = y_bin.shape[1]

# SVM Model with hyperparameters and sorted features
svm_features = [
    'stdA1', 'RMSSDppAmp', 'SDSDppAmp', 'SDppAmp', 'SDSDpw', 'stdT1', 'CCM', 'CCM_statistical', 'stdArea',
    'meanT2', 'AVpw', 'AVppAmp', 'meanA2', 'meanA1', 'RMSSDpw', 'shapeFactor', 'shapeFactor_statistical',
    'stdA2', 'MeanAbsDev', 'MeanAbsDev_statistical', 'SDpw', 'meanValue', 'meanValue_statistical', 'MedianAbsDev',
    'MedianAbsDev_statistical', 'InterquartileRange', 'InterquartileRange_statistical', 'stdT2', 'meanT1',
    'skewPPI', 'skewPPG', 'HjorthComplexity', 'HjorthComplexity_statistical', 'kurtPPI', 'kurtPPG', 'minValue',
    'minValue_statistical', 'HFD', 'HFD_statistical', 'PoincareSD2', 'PoincareSD2_statistical', 'PoincareSD1',
    'PoincareSD1_statistical', 'meanIPTR', 'meanArea', 'ratioSD1SD2', 'ratioSD1SD2_statistical', 'tmean50'
]

# RandomForest sorted features with hyperparameters
rf_features = [
    'AVpw', 'meanT1', 'HFD_statistical', 'HFD', 'meanA1', 'stdA1', 'stdIPAR', 'HjorthComplexity', 'meanArea',
    'HjorthComplexity_statistical', 'meanIPTR', 'meanA2', 'stdIPTR', 'meanT2', 'RMSSDppAmp', 'SDpw', 'meanIPAR',
    'stdT1', 'stdArea', 'SDSDppAmp', 'stdT2', 'meanValue', 'shapeFactor_statistical', 'shapeFactor', 'meanValue_statistical',
    'SDSDpw', 'SDppAmp', 'stdA2', 'skewPPG', 'tmean25', 'RMSSDpw', 'skewPPI', 'tmean25_statistical', 'AVppAmp',
    'geometricMean_statistical', 'minValue_statistical', 'minValue', 'tmean50_statistical', 'geometricMean',
    'centralMoment_statistical', 'InterquartileRange_statistical', 'tmean50', 'centralMoment', 'InterquartileRange',
    'MeanAbsDev_statistical', 'MedianAbsDev', 'MeanAbsDev', 'MedianAbsDev_statistical', 'kurtPPI', 'kurtPPG', 'PoincareSD2'
]

# CatBoost sorted features with hyperparameters
catboost_features = [
    'stdA1', 'RMSSDppAmp', 'HjorthComplexity', 'meanT2', 'stdIPAR', 'HjorthComplexity_statistical', 'SDpw', 'meanIPTR',
    'meanA2', 'AVpw', 'SDppAmp', 'stdIPTR', 'SDSDppAmp', 'shapeFactor_statistical', 'stdArea', 'tmean25', 'SDSDpw',
    'meanIPAR', 'shapeFactor', 'meanValue', 'meanT1', 'stdT1', 'stdT2', 'AVppAmp', 'HFD_statistical', 'meanValue_statistical',
    'HFD', 'meanArea', 'RMSSDpw', 'meanA1', 'stdA2', 'tmean25_statistical', 'tmean50', 'skewPPI', 'lam', 'tmean50_statistical',
    'MeanAbsDev', 'CCM', 'minValue', 'InterquartileRange', 'rmsValue', 'CCM_statistical'
]

# XGBoost sorted features with hyperparameters
xgb_features = [
    'AVpw', 'HjorthComplexity_statistical', 'meanT1', 'maxValue_statistical', 'stdIPTR', 'shapeFactor_statistical',
    'shapeFactor', 'stdA1', 'stdIPAR', 'ratioSD1SD2', 'HjorthComplexity', 'KFD_statistical', 'HFD', 'meanArea',
    'minValue_statistical', 'meanA2', 'SDpw', 'RMSSDppAmp', 'HFD_statistical', 'SDppAmp', 'MeanAbsDev', 'minValue',
    'MeanAbsDev_statistical', 'meanT2', 'PoincareSD1_statistical', 'tmean25', 'ratioSD1SD2_statistical', 'InterquartileRange_statistical',
    'maxValue', 'InterquartileRange', 'tmean25_statistical', 'KFD', 'HjorthMobility', 'MedianAbsDev_statistical',
    'AVppAmp', 'geometricMean', 'CCM', 'MedianAbsDev', 'meanIPTR', 'PoincareSD1', 'meanIPAR', 'CCM_statistical',
    'meanValue_statistical', 'skewPPG', 'meanA1', 'stdArea', 'centralMoment_statistical', 'SDSDpw', 'stdT2', 'PoincareSD2',
    'SDSDppAmp', 'meanValue', 'skewPPI', 'stdA2', 'stdT1', 'PoincareSD2_statistical', 'geometricMean_statistical'
]

# Hyperparameters for SVM
svm_params = {
    'tol': 0.12,
    'shrinking': False,
    'kernel': 'rbf',
    'gamma': 'scale',
    'degree': 3,
    'coef0': 0.5,
    'class_weight': 'balanced',
    'C': 100.0,
    'random_state': 160,
    'probability': True
}

# Hyperparameters for RandomForestClassifier
rf_params = {
    "n_estimators": 300,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "max_features": 'log2',
    "max_depth": 20,
    "criterion": 'entropy',
    "bootstrap": False,
    "random_state": 150
}

# Hyperparameters for CatBoostClassifier
catboost_params = {
    'loss_function': 'MultiClass', #added later
    'bagging_temperature': 0.41750608402789213,
    'boosting_type': 'Ordered',
    'bootstrap_type': 'MVS',
    'border_count': 65,
    'class_weights': {0: 0.155, 1: 0.319, 2: 0.150, 3: 0.376},
    'depth': 9,
    'grow_policy': 'SymmetricTree',
    'iterations': 858,
    'l2_leaf_reg': 2.8547,
    'leaf_estimation_iterations': 2,
    'leaf_estimation_method': 'Gradient',
    'learning_rate': 0.2898,
    'min_data_in_leaf': 7,
    'od_type': 'Iter',
    'od_wait': 38,
    'random_strength': 0.767,
    'rsm': 0.4617,
    'score_function': 'Cosine'
}

# Hyperparameters for XGBoost
xgb_params = {
    "objective": 'multi:softprob',  # changed from softmax to softprob to allow probability prediction
    "eval_metric": 'mlogloss',
    "random_state": 150,
    "subsample": 0.9,
    "reg_lambda": 0.04,
    "reg_alpha": 0.7,
    "n_estimators": 600,
    "max_depth": 9,
    "learning_rate": 0.05,
    "gamma": 0,
    "colsample_bytree": 0.9, #added later
    "num_class": 4
}

# Split data with stratification to ensure all classes are represented
def prepare_data(features):
    X = df[features]
    return train_test_split(X, y_bin, test_size=0.3, random_state=42, stratify=y)

# Function to fit and calculate macro-average ROC for a model
def calculate_macro_avg_roc(model, X_train, X_test, y_train, y_test):
    # Train the model and get predicted probabilities
    y_score = model.fit(X_train, y_train).predict_proba(X_test)

    # Convert y_score to NumPy array if it is not already
    y_score = np.array(y_score)

    # Handle incorrect shape: (4, 2829, 2) and reshape it to the correct format
    # Check if the shape indicates 3 dimensions (incorrect format)
    if len(y_score.shape) == 3:
        # Flatten or reshape as needed to get [n_samples, n_classes]
        y_score = np.concatenate([y_score[i] for i in range(y_score.shape[0])], axis=1)

    # Ensure y_test is binarized
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3])

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr["macro"], tpr["macro"], roc_auc["macro"]



# Prepare for plotting all models in one figure
plt.figure(figsize=(12, 10))

# Define colors and model names
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green']
model_names = ['SVM', 'Random Forest', 'CatBoost', 'XGBoost']

# SVM ROC
X_train, X_test, y_train, y_test = prepare_data(svm_features)
svm_model = OneVsRestClassifier(SVC(**svm_params))
fpr_svm, tpr_svm, roc_auc_svm = calculate_macro_avg_roc(svm_model, X_train, X_test, y_train, y_test)

# RandomForest ROC
X_train, X_test, y_train, y_test = prepare_data(rf_features)
rf_model = RandomForestClassifier(**rf_params)
fpr_rf, tpr_rf, roc_auc_rf = calculate_macro_avg_roc(rf_model, X_train, X_test, y_train, y_test)

# CatBoost ROC
X_train, X_test, y_train, y_test = train_test_split(df[catboost_features], y, test_size=0.3, random_state=42, stratify=y)

catboost_model = CatBoostClassifier(**catboost_params, verbose=0)
fpr_cat, tpr_cat, roc_auc_cat = calculate_macro_avg_roc(catboost_model, X_train, X_test, y_train, y_test)

# XGBoost ROC (no need to binarize y_train and y_test)
X_train, X_test, y_train, y_test = train_test_split(df[xgb_features], y, test_size=0.3, random_state=42, stratify=y)

# XGBoost model
xgb_model = XGBClassifier(**xgb_params)

# Calculate the ROC curve using the original multi-class labels
def calculate_macro_avg_roc_xgb(model, X_train, X_test, y_train, y_test):
    # Train model
    y_score = model.fit(X_train, y_train).predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Binarize y_test for ROC calculation (only for ROC purposes, not for model training)
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3])

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr["macro"], tpr["macro"], roc_auc["macro"]

# Use the corrected function for XGBoost
fpr_xgb, tpr_xgb, roc_auc_xgb = calculate_macro_avg_roc_xgb(xgb_model, X_train, X_test, y_train, y_test)


# Plot all ROC curves
plt.plot(fpr_svm, tpr_svm, color=colors[0], lw=2, label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot(fpr_rf, tpr_rf, color=colors[1], lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_cat, tpr_cat, color=colors[2], lw=2, label=f'CatBoost (AUC = {roc_auc_cat:.2f})')
plt.plot(fpr_xgb, tpr_xgb, color=colors[3], lw=2, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')

# Plot settings
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Macro-Averaged ROC Curves for All Models', fontsize=22)
plt.legend(loc="lower right", fontsize=12)

# Show the plot
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
