import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")
y = df["Class2"]

# New sorted features list
sorted_features = [
    'AVpw', 'meanT1', 'HFD_statistical', 'HFD', 'meanA1', 'stdA1', 'stdIPAR', 'HjorthComplexity', 'meanArea',
    'HjorthComplexity_statistical', 'meanIPTR', 'meanA2', 'stdIPTR', 'meanT2', 'RMSSDppAmp', 'SDpw', 'meanIPAR',
    'stdT1', 'stdArea', 'SDSDppAmp', 'stdT2', 'meanValue', 'shapeFactor_statistical', 'shapeFactor',
    'meanValue_statistical', 'SDSDpw', 'SDppAmp', 'stdA2', 'skewPPG', 'tmean25', 'RMSSDpw', 'skewPPI',
    'tmean25_statistical', 'AVppAmp', 'geometricMean_statistical', 'minValue_statistical', 'minValue',
    'tmean50_statistical', 'geometricMean', 'centralMoment_statistical', 'InterquartileRange_statistical',
    'tmean50', 'centralMoment', 'InterquartileRange', 'MeanAbsDev_statistical', 'MedianAbsDev', 'MeanAbsDev',
    'MedianAbsDev_statistical', 'kurtPPI', 'kurtPPG', 'PoincareSD2', 'maxValue', 'lam_statistical'
]

# Hyperparameters for the RandomForestClassifier
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

# Use all features for the model
X = df[sorted_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=160)

# Initialize and fit the RandomForest model
clf = RandomForestClassifier(**hyperparameters_RF)

# Measure training time
start_time = time.time()  # Start time
clf.fit(X_train, y_train)
end_time = time.time()  # End time
training_time = end_time - start_time  # Calculate training time

# Predict and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
accuracy_percentage = accuracy * 100  # Convert accuracy to percentage

# Output the result
print(f"Accuracy with all features: {accuracy:.4f}")
print(f"Training time: {training_time:.2f} seconds")  # Display training time

# Define the labels
labels = ['LS', 'DS', 'REM', 'Wake']

# Generate and print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labels))

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix with the 'Purples' colormap and integer format
plt.figure(figsize=(12, 10))  # Increase figure size to accommodate larger fonts
heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=True, annot_kws={"size": 35},
                      xticklabels=labels, yticklabels=labels)

# Customize the color bar font size to 25
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=25)

plt.xlabel('Predicted Label', fontsize=35)
plt.ylabel('True Label', fontsize=35)

# Include training time and accuracy percentage in the title
plt.title(f'Confusion Matrix\nTraining time: {training_time:.2f} seconds | Accuracy: {accuracy_percentage:.2f}%', fontsize=30)

plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

# Save the figure with bbox_inches set to tight
plt.savefig('confusion_matrix_RF.png', bbox_inches='tight')

# Show the plot
plt.show()
