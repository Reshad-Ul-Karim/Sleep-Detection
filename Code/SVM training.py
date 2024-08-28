

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset again
df = pd.read_csv("Sleep_Stage_Combo_main.csv")

# Prepare the data
#drop_columns = ['SubNo', "Class", "SegNo"]
drop_columns = ['SubNo', "SegNo", "Class",'averageTeagerEnergy', 'harmonicMean', 'svdPPI',
               'averageTeagerEnergy_statistical', 'harmonicMean_statistical', 'svdPPG']
X = df.drop(drop_columns, axis=1)
y = df["Class"]
rnd = 160
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rnd, stratify=y)

# Using the best parameters from the RandomizedSearchCV to train an SVM model
best_svm_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        tol=0.12, #0.12
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

# Train the model with the best parameters
best_svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred_best_svm = best_svm_model.predict(X_test)

# Evaluate the model
svm_accuracy = accuracy_score(y_test, y_pred_best_svm)
svm_class_report = classification_report(y_test, y_pred_best_svm)

print(svm_accuracy)
print(svm_class_report)


with open("SVM model_evaluation.txt", "w") as file:
    file.write(f'Accuracy for best SVM: {svm_accuracy * 100:.2f}%\n')
    file.write(svm_class_report)
