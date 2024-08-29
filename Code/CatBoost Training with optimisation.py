import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Sleep_Stage_Combo4.csv')

# Split the dataset into training and testing sets
drop_columns = ['SubNo', "SegNo", "Class", "Class2"]
X = df.drop(drop_columns, axis=1)
y = df["Class2"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

# Define the best parameters
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

# Train the CatBoostClassifier with the best parameters
model = CatBoostClassifier(
    **best_params,
    random_state=42,
    verbose=0
)

model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

#output file
output_file = open("CatBoost_model_evaluation.txt", "w")
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Test set accuracy: {accuracy * 100:.2f}%')
output_file.write(f'Test set accuracy: {accuracy * 100:.2f}%\n')
# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))
output_file.write(f"{classification_report(y_test, y_pred, zero_division=1)}")
# Generate and plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for CatBoost')
plt.savefig("catboost_confusion_matrix.png")  # Save the confusion matrix plot
plt.show()
