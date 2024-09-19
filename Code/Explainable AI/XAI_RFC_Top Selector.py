import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import defaultdict

sorted_features = ['AVpw', 'meanT1', 'HFD_statistical', 'HFD', 'meanA1', 'stdA1', 'stdIPAR', 'HjorthComplexity',
                   'meanArea',
                   'HjorthComplexity_statistical', 'meanIPTR', 'meanA2', 'stdIPTR', 'meanT2', 'RMSSDppAmp', 'SDpw',
                   'meanIPAR', 'stdT1',
                   'stdArea', 'SDSDppAmp', 'stdT2', 'meanValue', 'shapeFactor_statistical', 'shapeFactor',
                   'meanValue_statistical',
                   'SDSDpw', 'SDppAmp', 'stdA2', 'skewPPG', 'tmean25', 'RMSSDpw', 'skewPPI', 'tmean25_statistical',
                   'AVppAmp',
                   'geometricMean_statistical', 'minValue_statistical', 'minValue', 'tmean50_statistical',
                   'geometricMean',
                   'centralMoment_statistical', 'InterquartileRange_statistical', 'tmean50', 'centralMoment',
                   'InterquartileRange',
                   'MeanAbsDev_statistical', 'MedianAbsDev', 'MeanAbsDev', 'MedianAbsDev_statistical', 'kurtPPI',
                   'kurtPPG',
                   'PoincareSD2', 'maxValue', 'lam_statistical', 'PoincareSD2_statistical', 'lam',
                   'HjorthMobility_statistical',
                   'maxValue_statistical', 'ratioSD1SD2_statistical', 'PoincareSD1', 'PoincareSD1_statistical',
                   'ratioSD1SD2',
                   'HjorthMobility', 'KFD', 'KFD_statistical', 'rmsValue', 'sdValue_statistical',
                   'rmsValue_statistical', 'sdValue',
                   'CCM', 'CCM_statistical', 'averageCurveLength', 'averageCurveLength_statistical']
df = pd.read_csv("Sleep_Stage_Combo2.csv")
y = df["Class2"]
sorted_acc = defaultdict(float)
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
for i in range(1, len(sorted_features)):
    X = df[sorted_features[:i]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(**hyperparameters_RF)
    clf.fit(X_train, y_train)
    print(f"Top {i} features: {clf.score(X_test, y_test)}")
    sorted_acc[i] = clf.score(X_test, y_test)
sorted_acc = dict(sorted(sorted_acc.items(), key=lambda x: x[1], reverse=True))
print(print(f"Top {list(sorted_acc.keys())[0]} features: {list(sorted_acc.values())[0]}"))