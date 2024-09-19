import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from collections import defaultdict

# List of sorted features
sorted_features = ['stdA1', 'RMSSDppAmp', 'HjorthComplexity', 'meanT2', 'stdIPAR', 'HjorthComplexity_statistical',
                   'SDpw', 'meanIPTR', 'meanA2', 'AVpw', 'SDppAmp', 'stdIPTR', 'SDSDppAmp', 'shapeFactor_statistical',
                   'stdArea', 'tmean25', 'SDSDpw', 'meanIPAR', 'shapeFactor', 'meanValue', 'meanT1', 'stdT1', 'stdT2',
                   'AVppAmp', 'HFD_statistical', 'meanValue_statistical', 'HFD', 'meanArea', 'RMSSDpw', 'meanA1',
                   'stdA2', 'tmean25_statistical', 'tmean50', 'skewPPI', 'lam', 'tmean50_statistical', 'MeanAbsDev',
                   'CCM', 'minValue', 'InterquartileRange', 'rmsValue', 'CCM_statistical', 'geometricMean',
                   'centralMoment', 'skewPPG', 'InterquartileRange_statistical', 'minValue_statistical', 'MedianAbsDev',
                   'MedianAbsDev_statistical', 'rmsValue_statistical', 'kurtPPI', 'lam_statistical',
                   'averageCurveLength',
                   'centralMoment_statistical', 'MeanAbsDev_statistical', 'sdValue', 'geometricMean_statistical',
                   'PoincareSD2', 'PoincareSD2_statistical', 'maxValue', 'PoincareSD1_statistical', 'kurtPPG',
                   'maxValue_statistical', 'HjorthMobility', 'averageCurveLength_statistical', 'sdValue_statistical',
                   'ratioSD1SD2_statistical', 'KFD_statistical', 'HjorthMobility_statistical', 'PoincareSD1', 'KFD',
                   'ratioSD1SD2']

# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")
y = df["Class2"]
sorted_acc = defaultdict(float)

# CatBoost hyperparameters
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

for i in range(1, len(sorted_features) + 1):
    X = df[sorted_features[:i]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize CatBoost classifier
    clf = CatBoostClassifier(**best_params, verbose=0)

    # Fit the model
    clf.fit(X_train, y_train)

    # Evaluate the model
    accuracy = clf.score(X_test, y_test)
    print(f"Top {i} features: {accuracy:.4f}")
    sorted_acc[i] = accuracy

# Sort the accuracies and print the top-performing feature count
sorted_acc = dict(sorted(sorted_acc.items(), key=lambda x: x[1], reverse=True))
top_features_count = list(sorted_acc.keys())[0]
top_accuracy = list(sorted_acc.values())[0]

# Get the actual top features
best_features = sorted_features[:top_features_count]

print(f"Top {top_features_count} features produce the best accuracy of {top_accuracy:.4f}:")
print(best_features)
