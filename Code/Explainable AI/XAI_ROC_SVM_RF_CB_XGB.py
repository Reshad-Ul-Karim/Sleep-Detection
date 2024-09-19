import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import numpy as np

# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")
# Split the dataset into features (X) and labels (y)
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
# Split the dataset into training and testing sets

# SVM Model with hyperparameters and sorted features
Features = {"SVM": [
    'stdA1', 'RMSSDppAmp', 'SDSDppAmp', 'SDppAmp', 'SDSDpw', 'stdT1', 'CCM', 'CCM_statistical', 'stdArea',
    'meanT2', 'AVpw', 'AVppAmp', 'meanA2', 'meanA1', 'RMSSDpw', 'shapeFactor', 'shapeFactor_statistical',
    'stdA2', 'MeanAbsDev', 'MeanAbsDev_statistical', 'SDpw', 'meanValue', 'meanValue_statistical', 'MedianAbsDev',
    'MedianAbsDev_statistical', 'InterquartileRange', 'InterquartileRange_statistical', 'stdT2', 'meanT1',
    'skewPPI', 'skewPPG', 'HjorthComplexity', 'HjorthComplexity_statistical', 'kurtPPI', 'kurtPPG', 'minValue',
    'minValue_statistical', 'HFD', 'HFD_statistical', 'PoincareSD2', 'PoincareSD2_statistical', 'PoincareSD1',
    'PoincareSD1_statistical', 'meanIPTR', 'meanArea', 'ratioSD1SD2', 'ratioSD1SD2_statistical', 'tmean50'
],
    "Random Forest": [
        'AVpw', 'meanT1', 'HFD_statistical', 'HFD', 'meanA1', 'stdA1', 'stdIPAR', 'HjorthComplexity', 'meanArea',
        'HjorthComplexity_statistical', 'meanIPTR', 'meanA2', 'stdIPTR', 'meanT2', 'RMSSDppAmp', 'SDpw', 'meanIPAR',
        'stdT1', 'stdArea', 'SDSDppAmp', 'stdT2', 'meanValue', 'shapeFactor_statistical', 'shapeFactor',
        'meanValue_statistical',
        'SDSDpw', 'SDppAmp', 'stdA2', 'skewPPG', 'tmean25', 'RMSSDpw', 'skewPPI', 'tmean25_statistical', 'AVppAmp',
        'geometricMean_statistical', 'minValue_statistical', 'minValue', 'tmean50_statistical', 'geometricMean',
        'centralMoment_statistical', 'InterquartileRange_statistical', 'tmean50', 'centralMoment', 'InterquartileRange',
        'MeanAbsDev_statistical', 'MedianAbsDev', 'MeanAbsDev', 'MedianAbsDev_statistical', 'kurtPPI', 'kurtPPG',
        'PoincareSD2'
    ],
    "CatBoost": [
        'stdA1', 'RMSSDppAmp', 'HjorthComplexity', 'meanT2', 'stdIPAR', 'HjorthComplexity_statistical', 'SDpw',
        'meanIPTR',
        'meanA2', 'AVpw', 'SDppAmp', 'stdIPTR', 'SDSDppAmp', 'shapeFactor_statistical', 'stdArea', 'tmean25', 'SDSDpw',
        'meanIPAR', 'shapeFactor', 'meanValue', 'meanT1', 'stdT1', 'stdT2', 'AVppAmp', 'HFD_statistical',
        'meanValue_statistical',
        'HFD', 'meanArea', 'RMSSDpw', 'meanA1', 'stdA2', 'tmean25_statistical', 'tmean50', 'skewPPI', 'lam',
        'tmean50_statistical',
        'MeanAbsDev', 'CCM', 'minValue', 'InterquartileRange', 'rmsValue', 'CCM_statistical'
    ],
    "XGBoost": [
    'AVpw', 'HjorthComplexity_statistical', 'meanT1', 'maxValue_statistical', 'stdIPTR', 'shapeFactor_statistical',
    'shapeFactor', 'stdA1', 'stdIPAR', 'ratioSD1SD2', 'HjorthComplexity', 'KFD_statistical', 'HFD', 'meanArea',
    'minValue_statistical', 'meanA2', 'SDpw', 'RMSSDppAmp', 'HFD_statistical', 'SDppAmp', 'MeanAbsDev', 'minValue',
    'MeanAbsDev_statistical', 'meanT2', 'PoincareSD1_statistical', 'tmean25', 'ratioSD1SD2_statistical',
    'InterquartileRange_statistical', 'maxValue', 'InterquartileRange', 'tmean25_statistical', 'KFD', 'HjorthMobility',
    'MedianAbsDev_statistical', 'AVppAmp', 'geometricMean', 'CCM', 'MedianAbsDev', 'meanIPTR', 'PoincareSD1',
    'meanIPAR', 'CCM_statistical', 'meanValue_statistical', 'skewPPG', 'meanA1', 'stdArea', 'centralMoment_statistical',
    'SDSDpw', 'stdT2', 'PoincareSD2', 'SDSDppAmp', 'meanValue', 'skewPPI', 'stdA2', 'stdT1', 'PoincareSD2_statistical',
]
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
        random_state=160,
        probability=True  # Enable probability estimates for ROC curve

    ))
])

# cnn model not GBM
# Initialize and train the RFC, SVM, and XGBoost models
models = [
    ("XGBoost", xgb, "Blue"),
    ("Random Forest", rf, "Purple"),
    ("CatBoost", cb, "Red"),
    ("SVM", svm, "Orange")
]

drop_columns = ['SubNo', "SegNo", "Class", "Class2", 'averageTeagerEnergy', 'harmonicMean', 'svdPPI',
                'averageTeagerEnergy_statistical', 'harmonicMean_statistical', 'svdPPG']
# train test split
plt.figure(figsize=(10, 8))
acc_score = {name: 0 for name, model, color in models}
for name, model, color in models:
    y = df["Class2"]
    X = df[Features[name]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    acc_score[name] = accuracy
    print(f'Accuracy for {name}: {accuracy * 100:.2f}%')
    print(classification_report(y_test, y_pred, zero_division=1))

    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    n_classes = y_test_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_pred_proba = model.predict_proba(X_test)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.plot(fpr["micro"], tpr["micro"],
             label=f'{name} (micro-average ROC curve) (area = {roc_auc["micro"]:.2f})',
             color=color, linestyle=':', linewidth=4)

# Plot the diagonal line
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('ROC_Curve.png', bbox_inches='tight')
plt.show()
