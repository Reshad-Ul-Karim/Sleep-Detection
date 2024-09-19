import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from collections import defaultdict

sorted_features = ['AVpw', 'HjorthComplexity_statistical', 'meanT1', 'maxValue_statistical', 'stdIPTR',
                   'shapeFactor_statistical', 'shapeFactor', 'stdA1', 'stdIPAR', 'ratioSD1SD2', 'HjorthComplexity',
                   'KFD_statistical', 'HFD', 'meanArea', 'minValue_statistical', 'meanA2', 'SDpw', 'RMSSDppAmp',
                   'HFD_statistical', 'SDppAmp', 'MeanAbsDev', 'minValue', 'MeanAbsDev_statistical', 'meanT2',
                   'PoincareSD1_statistical', 'tmean25', 'ratioSD1SD2_statistical', 'InterquartileRange_statistical',
                   'maxValue', 'InterquartileRange', 'tmean25_statistical', 'KFD', 'HjorthMobility',
                   'MedianAbsDev_statistical', 'AVppAmp', 'geometricMean', 'CCM', 'MedianAbsDev', 'meanIPTR',
                   'PoincareSD1', 'meanIPAR', 'CCM_statistical', 'meanValue_statistical', 'skewPPG', 'meanA1',
                   'stdArea', 'centralMoment_statistical', 'SDSDpw', 'stdT2', 'PoincareSD2', 'SDSDppAmp', 'meanValue',
                   'skewPPI', 'stdA2', 'stdT1', 'PoincareSD2_statistical', 'geometricMean_statistical', 'centralMoment',
                   'HjorthMobility_statistical', 'tmean50', 'kurtPPI', 'RMSSDpw', 'tmean50_statistical',
                   'lam_statistical', 'kurtPPG', 'lam', 'averageCurveLength_statistical', 'rmsValue', 'sdValue',
                   'averageCurveLength', 'rmsValue_statistical', 'sdValue_statistical']

df = pd.read_csv("Sleep_Stage_Combo2.csv")
y = df["Class2"]
sorted_acc = defaultdict(float)
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
for i in range(1, len(sorted_features)):
    X = df[sorted_features[:i]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = XGBClassifier(**hyperparameters_XGB)
    clf.fit(X_train, y_train)
    print(f"Top {i} features: {clf.score(X_test, y_test)}")
    sorted_acc[i] = clf.score(X_test, y_test)
sorted_acc = dict(sorted(sorted_acc.items(), key=lambda x: x[1], reverse=True))
print(print(f"Top {list(sorted_acc.keys())[0]} features: {list(sorted_acc.values())[0]}"))