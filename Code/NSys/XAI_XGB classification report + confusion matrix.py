import time

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# New sorted features list
sorted_features = [
    'AVpw', 'HjorthComplexity_statistical', 'meanT1', 'maxValue_statistical', 'stdIPTR', 'shapeFactor_statistical',
    'shapeFactor', 'stdA1', 'stdIPAR', 'ratioSD1SD2', 'HjorthComplexity', 'KFD_statistical', 'HFD', 'meanArea',
    'minValue_statistical', 'meanA2', 'SDpw', 'RMSSDppAmp', 'HFD_statistical', 'SDppAmp', 'MeanAbsDev', 'minValue',
    'MeanAbsDev_statistical', 'meanT2', 'PoincareSD1_statistical', 'tmean25', 'ratioSD1SD2_statistical',
    'InterquartileRange_statistical', 'maxValue', 'InterquartileRange', 'tmean25_statistical', 'KFD', 'HjorthMobility',
    'MedianAbsDev_statistical', 'AVppAmp', 'geometricMean', 'CCM', 'MedianAbsDev', 'meanIPTR', 'PoincareSD1',
    'meanIPAR', 'CCM_statistical', 'meanValue_statistical', 'skewPPG', 'meanA1', 'stdArea', 'centralMoment_statistical',
    'SDSDpw', 'stdT2', 'PoincareSD2', 'SDSDppAmp', 'meanValue', 'skewPPI', 'stdA2', 'stdT1', 'PoincareSD2_statistical',
]

# Load dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")
y = df["Class2"]

# Hyperparameters for the XGBClassifier
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

# Use all features for the model
X = df[sorted_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Measure the start time
start_time = time.time()

# Initialize and fit the XGBoost model
clf = XGBClassifier(**hyperparameters_XGB)
clf.fit(X_train, y_train)

# Measure the end time
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

# Predict and calculate accuracy
y_pred = clf.predict(X_test)

# Output the result
accuracy = clf.score(X_test, y_test)
print(f"Accuracy with all features: {accuracy:.4f}")

# Define the labels
labels = ['LS', 'DS', 'REM', 'Wake']

# Generate and print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labels))

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix with the 'Greens' colormap
plt.figure(figsize=(12, 10))  # Increase figure size to accommodate larger fonts
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=True, annot_kws={"size": 35},
            xticklabels=labels, yticklabels=labels[::-1])  # Use labels for ticks
plt.xlabel('Predicted Label', fontsize=35)
plt.ylabel('True Label', fontsize=35)
plt.title('Confusion Matrix', fontsize=40)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
# Include training time and accuracy percentage in the title
plt.title(f'Confusion Matrix\nTraining time: {training_time:.2f} seconds | Accuracy: {accuracy*100:.2f}%', fontsize=30)

# Save and display the plot
plt.savefig('confusion_matrix_xgb_label.png', bbox_inches='tight')  # Save the figure with tight bounding box
plt.show()
