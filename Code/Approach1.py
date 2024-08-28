import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")

# Split the dataset into training and testing sets
drop_columns = ['SubNo', "SegNo", "Class", "Class2", 'averageTeagerEnergy', 'harmonicMean', 'svdPPI',
       'averageTeagerEnergy_statistical', 'harmonicMean_statistical','svdPPG']
X = df.drop(drop_columns, axis=1)
y = df["Class2"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=150)


# Initialize and train the RFC and XGBoost models
cb = CatBoostClassifier()
xgb = XGBClassifier()
rf = RandomForestClassifier(n_estimators=100, random_state=150)

models = [
    ("XGBoost", xgb),
    ("Random Forest", rf),
    ("CatBoost", cb),

]

acc_score = {name: 0 for name, model in models}

output_file = open("Approach 1 model evaluation.txt", "w")

for name, model in models:
    model.fit(X_train, y_train)
for name, model in models:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    acc_score[name] = accuracy
    print(f'Accuracy for {name}: {accuracy * 100:.2f}%')
    print(classification_report(y_test, y_pred, zero_division=1))
    output_file.write(f'Accuracy for {name}: {accuracy * 100:.2f}%\n')
    output_file.write(classification_report(y_test, y_pred, zero_division=1))
