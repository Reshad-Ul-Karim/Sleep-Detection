import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo_main.csv")

# Split the dataset into training and testing sets
drop_columns = ['SubNo', "SegNo", "Class",'averageTeagerEnergy', 'harmonicMean', 'svdPPI',
                'averageTeagerEnergy_statistical', 'harmonicMean_statistical', 'svdPPG']
X = df.drop(drop_columns, axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=150)

# Initialize the XGBClassifier with the best parameters
best_xgb = XGBClassifier(
    objective='multi:softmax',
    eval_metric='mlogloss',
    random_state=150,
    subsample=0.9,
    reg_lambda=0.04,
    reg_alpha=0.7,
    n_estimators=600,
    max_depth=9,
    learning_rate=0.05,
    gamma=0,
    colsample_bytree=0.9
)

# Fit the model on the training data
best_xgb.fit(X_train, y_train)

# Predict on the test set
y_pred = best_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy and classification report
print(f'Accuracy for best XGBoost: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred, zero_division=1))

# Save the evaluation results to a file
with open("XGBoost_model_evaluation2.txt", "w") as file:
    file.write(f'Accuracy for best XGBoost: {accuracy * 100:.2f}%\n')
    file.write(classification_report(y_test, y_pred, zero_division=1))
