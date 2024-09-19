import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")
y = df["Class2"]

# Updated sorted features list
sorted_features = [
    'stdA1', 'RMSSDppAmp', 'SDSDppAmp', 'SDppAmp', 'SDSDpw', 'stdT1', 'CCM', 'CCM_statistical', 'stdArea',
    'meanT2', 'AVpw', 'AVppAmp', 'meanA2', 'meanA1', 'RMSSDpw', 'shapeFactor', 'shapeFactor_statistical',
    'stdA2', 'MeanAbsDev', 'MeanAbsDev_statistical', 'SDpw', 'meanValue', 'meanValue_statistical', 'MedianAbsDev',
    'MedianAbsDev_statistical', 'InterquartileRange', 'InterquartileRange_statistical', 'stdT2', 'meanT1',
    'skewPPI', 'skewPPG', 'HjorthComplexity', 'HjorthComplexity_statistical', 'kurtPPI', 'kurtPPG', 'minValue',
    'minValue_statistical', 'HFD', 'HFD_statistical', 'PoincareSD2', 'PoincareSD2_statistical', 'PoincareSD1',
    'PoincareSD1_statistical', 'meanIPTR', 'meanArea', 'ratioSD1SD2', 'ratioSD1SD2_statistical', 'tmean50'
]

# Define the SVM model with the specified parameters
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

# Use all features for the model
X = df[sorted_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=160, stratify=y)

# Measure training time
start_time = time.time()  # Start time
# Fit the model using all features
svm_model.fit(X_train, y_train)
end_time = time.time()  # End time
training_time = end_time - start_time  # Calculate training time

# Predict and calculate accuracy
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy_percentage = accuracy * 100  # Convert accuracy to percentage

# Output the accuracy
print(f"Accuracy with all features: {accuracy:.4f}")
print(f"Training time: {training_time:.2f} seconds")  # Display training time

# Define the labels
labels = ['LS', 'DS', 'REM', 'Wake']

# Generate and print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labels))

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix with larger fonts
plt.figure(figsize=(12, 10))  # Increase figure size to accommodate larger fonts
heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 35},
                      xticklabels=labels[::-1], yticklabels=labels[::-1])  # Use labels for ticks

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
plt.savefig('confusion_matrix_svm.png', bbox_inches='tight')

# Show the plot
plt.show()
