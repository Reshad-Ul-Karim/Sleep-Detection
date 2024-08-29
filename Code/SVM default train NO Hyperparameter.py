import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")

# Prepare the data
drop_columns = ['SubNo', "SegNo", "Class", "Class2", 'averageTeagerEnergy', 'harmonicMean', 'svdPPI',
                'averageTeagerEnergy_statistical', 'harmonicMean_statistical', 'svdPPG']
X = df.drop(drop_columns, axis=1)
y = df["Class2"]

# Split the data into training and testing sets
rnd = 160
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rnd, stratify=y)

# Create an SVM model with default hyperparameters
default_svm_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=rnd))
])

# Train the model with the default parameters
default_svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred_default_svm = default_svm_model.predict(X_test)

# Evaluate the model
default_svm_accuracy = accuracy_score(y_test, y_pred_default_svm)
default_svm_class_report = classification_report(y_test, y_pred_default_svm)

print(f'Accuracy with default SVM: {default_svm_accuracy * 100:.2f}%')
print(default_svm_class_report)


# Save the evaluation results to a text file
#with open("Default_SVM_model_evaluation.txt", "w") as file:
#    file.write(f'Accuracy for default SVM: {default_svm_accuracy * 100:.2f}%\n')