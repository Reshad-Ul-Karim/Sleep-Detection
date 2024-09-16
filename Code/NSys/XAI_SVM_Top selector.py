import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import defaultdict

# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")
y = df["Class2"]

# Sorted features based on your manual sorting

sorted_features = [
    'stdA1', 'RMSSDppAmp', 'SDSDppAmp', 'SDppAmp', 'SDSDpw', 'stdT1', 'CCM', 'CCM_statistical', 'stdArea',
    'meanT2', 'AVpw', 'AVppAmp', 'meanA2', 'meanA1', 'RMSSDpw', 'shapeFactor', 'shapeFactor_statistical',
    'stdA2', 'MeanAbsDev', 'MeanAbsDev_statistical', 'SDpw', 'meanValue', 'meanValue_statistical', 'MedianAbsDev',
    'MedianAbsDev_statistical', 'InterquartileRange', 'InterquartileRange_statistical', 'stdT2', 'meanT1',
    'skewPPI', 'skewPPG', 'HjorthComplexity', 'HjorthComplexity_statistical', 'kurtPPI', 'kurtPPG', 'minValue',
    'minValue_statistical', 'HFD', 'HFD_statistical', 'PoincareSD2', 'PoincareSD2_statistical', 'PoincareSD1',
    'PoincareSD1_statistical', 'meanIPTR', 'meanArea', 'ratioSD1SD2', 'ratioSD1SD2_statistical', 'tmean50',
    'tmean50_statistical', 'stdIPTR', 'tmean25', 'tmean25_statistical', 'geometricMean', 'geometricMean_statistical',
    'lam', 'lam_statistical', 'HjorthMobility', 'HjorthMobility_statistical', 'averageCurveLength',
    'averageCurveLength_statistical', 'maxValue', 'maxValue_statistical', 'meanIPAR', 'centralMoment',
    'centralMoment_statistical', 'stdIPAR', 'KFD', 'KFD_statistical', 'rmsValue', 'rmsValue_statistical',
    'sdValue', 'sdValue_statistical'
]



# Initialize an empty dictionary to store accuracy scores for top features
sorted_acc_svm = defaultdict(float)

# Define the best SVM model (from your previous pipeline)
rnd = 160
svm_model = Pipeline([
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
        random_state=rnd
    ))
])

# Loop through each top n feature set and evaluate accuracy
for i in range(1, len(sorted_features) + 1):
    # Use the top i features
    X = df[sorted_features[:i]]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Fit the model with top i features
    svm_model.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Store the accuracy for the current number of top features
    sorted_acc_svm[i] = accuracy

    # Print the result for the current number of top features
    print(f"Top {i} features accuracy: {accuracy:.4f}")

# Sort the accuracy dictionary by value (descending order)
sorted_acc_svm = dict(sorted(sorted_acc_svm.items(), key=lambda x: x[1], reverse=True))

# Output the best feature set and corresponding accuracy
best_num_features = list(sorted_acc_svm.keys())[0]
best_accuracy = list(sorted_acc_svm.values())[0]

print(f"\nBest result: Top {best_num_features} features with accuracy: {best_accuracy:.4f}")
