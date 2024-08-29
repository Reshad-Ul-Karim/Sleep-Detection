import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")

# Split the dataset into training and testing sets
drop_columns = ['SubNo', "SegNo", "Class", "Class2", 'averageTeagerEnergy', 'harmonicMean', 'svdPPI',
                'averageTeagerEnergy_statistical', 'harmonicMean_statistical', 'svdPPG']
X = df.drop(drop_columns, axis=1)
y = df["Class2"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=150)

# Train the RandomForestClassifier with the best parameters
best_rf = RandomForestClassifier(
    n_estimators=600,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=20,
    criterion='entropy',
    bootstrap=False,
    random_state=90
)

best_rf.fit(X_train, y_train)

# Predict on the test set
y_pred = best_rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy for best Random Forest: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred, zero_division=1))

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=best_rf.classes_, yticklabels=best_rf.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Random Forest')

# Save the plot
plt.savefig("confusion_matrix_rf.png")  # Save the plot as a .png file

# Show the plot
plt.show()
