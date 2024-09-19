import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")
y = df["Class2"]

# Top 34 features based on your manual sorting
top_34_features = [
    'stdA1', 'RMSSDppAmp', 'SDSDppAmp', 'SDppAmp', 'SDSDpw', 'stdT1', 'CCM', 'CCM_statistical', 'stdArea',
    'meanT2', 'AVpw', 'AVppAmp', 'meanA2', 'meanA1', 'RMSSDpw', 'shapeFactor', 'shapeFactor_statistical',
    'stdA2', 'MeanAbsDev', 'MeanAbsDev_statistical', 'SDpw', 'meanValue', 'meanValue_statistical', 'MedianAbsDev',
    'MedianAbsDev_statistical', 'InterquartileRange', 'InterquartileRange_statistical', 'stdT2', 'meanT1',
    'skewPPI', 'skewPPG', 'HjorthComplexity', 'HjorthComplexity_statistical', 'kurtPPI', 'kurtPPG'
]

# Subset the data using the top 34 features
X = df[top_34_features]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=160, stratify=y)

# Use the best hyperparameters you found
best_params = {
    'C': 37.163,
    'coef0': 0.9173551369131957,
    'degree': 4,
    'gamma': 0.06631797623183432,
    'kernel': 'rbf'
}

# Define the SVM pipeline with the best hyperparameters
svm_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        C=best_params['C'],
        coef0=best_params['coef0'],
        degree=best_params['degree'],
        gamma=best_params['gamma'],
        kernel=best_params['kernel'],
        class_weight='balanced',
        random_state=160
    ))
])

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy with best hyperparameters SVM: {accuracy*100:.2f}")
print(f"Classification Report:\n{class_report}")

# Save the results to a text file
with open("XAI_SVM_Final_Model_Results.txt", "w") as output_file:
    output_file.write(f"Accuracy with best hyperparameters: {accuracy:.4f}\n")
    output_file.write(f"Classification Report:\n{class_report}\n")

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.set(font_scale=1.5)  # Set font size for better visibility
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 30})
plt.title('Confusion Matrix', fontsize=22)
plt.xlabel('Predicted Label', fontsize=22)
plt.ylabel('True Label', fontsize=22)
plt.tight_layout()

# Save the confusion matrix plot
plt.savefig('SVM_Confusion_Matrix.png', bbox_inches='tight')

# Show the plot
plt.show()
