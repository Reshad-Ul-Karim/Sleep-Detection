import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time  # Import the time module to measure training time

# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")
y = df["Class2"]

# Use the first 42 features from the sorted features list
sorted_features = ['stdA1', 'RMSSDppAmp', 'HjorthComplexity', 'meanT2', 'stdIPAR', 'HjorthComplexity_statistical',
                   'SDpw', 'meanIPTR', 'meanA2', 'AVpw', 'SDppAmp', 'stdIPTR', 'SDSDppAmp', 'shapeFactor_statistical',
                   'stdArea', 'tmean25', 'SDSDpw', 'meanIPAR', 'shapeFactor', 'meanValue', 'meanT1', 'stdT1',
                   'stdT2', 'AVppAmp', 'HFD_statistical', 'meanValue_statistical', 'HFD', 'meanArea', 'RMSSDpw',
                   'meanA1', 'stdA2', 'tmean25_statistical', 'tmean50', 'skewPPI', 'lam', 'tmean50_statistical',
                   'MeanAbsDev', 'CCM', 'minValue', 'InterquartileRange', 'rmsValue', 'CCM_statistical']

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

# Use the first 42 features for the model
X = df[sorted_features[:42]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Measure the start time
start_time = time.time()

# Initialize and train the CatBoost classifier
catboost_model = CatBoostClassifier(**best_params, verbose=0)
catboost_model.fit(X_train, y_train)

# Measure the end time
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

# Predict and calculate accuracy
y_pred = catboost_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy_percentage = accuracy * 100

# Output the accuracy
print(f"Accuracy with the first 42 features: {accuracy:.4f}")

# Generate and print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Adjust the labels in the confusion matrix plot according to the provided mapping
label_mapping = {0: 'LS', 1: 'DS', 2: 'REM', 3: 'Wake'}

# Plot the confusion matrix with the correct label mapping for both axes
plt.figure(figsize=(12, 10))  # Increase figure size to accommodate larger fonts
heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=True, annot_kws={"size": 35},
                      xticklabels=[label_mapping[i] for i in range(4)][::-1],
                      yticklabels=[label_mapping[i] for i in range(4)][::-1])   # Correct order of y-axis labels

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
plt.savefig('confusion_matrix_catboost.png', bbox_inches='tight')

# Show the plot
plt.show()


