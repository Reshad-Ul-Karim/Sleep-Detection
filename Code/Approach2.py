import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


# Load the dataset
df = pd.read_csv("PPG_Train_Data1.csv")
df2 = pd.read_csv("PPG_Test_Data1.csv")


# Split the dataset into training and testing sets
drop_columns = ['SubNo', "SegNo", "Class", 'averageTeagerEnergy', 'harmonicMean', 'svdPPI',
       'averageTeagerEnergy_statistical', 'harmonicMean_statistical','svdPPG']
X_train = df.drop(drop_columns, axis=1)
y_train = df["Class"]
X_test = df2.drop(drop_columns, axis=1)
y_test = df2["Class"]




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

for name, model in models:
    model.fit(X_train, y_train)
for name, model in models:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    acc_score[name] = accuracy
    print(f'Accuracy for {name}: {accuracy * 100:.2f}%')
    print(classification_report(y_test, y_pred, zero_division=1))
