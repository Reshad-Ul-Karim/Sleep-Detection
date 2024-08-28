import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# List of file paths for each iteration
train_files = [
    r"E:\Paper\Brain signal\train_approach3\Sleep_Stage_subject_TRAIN_1.csv",
    r"E:\Paper\Brain signal\train_approach3\Sleep_Stage_subject_TRAIN_2.csv",
    r"E:\Paper\Brain signal\train_approach3\Sleep_Stage_subject_TRAIN_3.csv",
    r"E:\Paper\Brain signal\train_approach3\Sleep_Stage_subject_TRAIN_4.csv",
    r"E:\Paper\Brain signal\train_approach3\Sleep_Stage_subject_TRAIN_5.csv",
    r"E:\Paper\Brain signal\train_approach3\Sleep_Stage_subject_TRAIN_6.csv",
    r"E:\Paper\Brain signal\train_approach3\Sleep_Stage_subject_TRAIN_7.csv",
    r"E:\Paper\Brain signal\train_approach3\Sleep_Stage_subject_TRAIN_8.csv",
    r"E:\Paper\Brain signal\train_approach3\Sleep_Stage_subject_TRAIN_9.csv",
    r"E:\Paper\Brain signal\train_approach3\Sleep_Stage_subject_TRAIN_10.csv",
]

test_files = [
    r"E:\Paper\Brain signal\test_approach3\Sleep_Stage_subject_TEST_1.csv",
    r"E:\Paper\Brain signal\test_approach3\Sleep_Stage_subject_TEST_2.csv",
    r"E:\Paper\Brain signal\test_approach3\Sleep_Stage_subject_TEST_3.csv",
    r"E:\Paper\Brain signal\test_approach3\Sleep_Stage_subject_TEST_4.csv",
    r"E:\Paper\Brain signal\test_approach3\Sleep_Stage_subject_TEST_5.csv",
    r"E:\Paper\Brain signal\test_approach3\Sleep_Stage_subject_TEST_6.csv",
    r"E:\Paper\Brain signal\test_approach3\Sleep_Stage_subject_TEST_7.csv",
    r"E:\Paper\Brain signal\test_approach3\Sleep_Stage_subject_TEST_8.csv",
    r"E:\Paper\Brain signal\test_approach3\Sleep_Stage_subject_TEST_9.csv",
    r"E:\Paper\Brain signal\test_approach3\Sleep_Stage_subject_TEST_10.csv",
]

# Dictionary to store cumulative accuracy scores for each model
cumulative_acc_score = {name: 0 for name in ["XGBoost", "Random Forest", "CatBoost"]}
total_iterations = len(train_files)

# Loop over each pair of train and test files
for i, (train_file, test_file) in enumerate(zip(train_files, test_files), start=1):
    print(f"Iteration {i}:")

    # Load the dataset
    df = pd.read_csv(train_file)
    df2 = pd.read_csv(test_file)

    # Split the dataset into features (X) and labels (y)
    drop_columns = ['SubNo', "SegNo", "Class", 'averageTeagerEnergy', 'harmonicMean', 'svdPPI',
                    'averageTeagerEnergy_statistical', 'harmonicMean_statistical', 'svdPPG']
    X_train = df.drop(drop_columns, axis=1)
    y_train = df["Class"]
    X_test = df2.drop(drop_columns, axis=1)
    y_test = df2["Class"]

    # Initialize the models
    cb = CatBoostClassifier(verbose=0)
    xgb = XGBClassifier()
    rf = RandomForestClassifier(n_estimators=100, random_state=150)

    models = [
        ("XGBoost", xgb),
        ("Random Forest", rf),
        ("CatBoost", cb),
    ]

    # Dictionary to store accuracy scores for the current iteration
    acc_score = {name: 0 for name, model in models}

    # Train and evaluate each model
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        acc_score[name] = accuracy
        cumulative_acc_score[name] += accuracy
        print(f'Accuracy for {name}: {accuracy * 100:.2f}%')
        print(classification_report(y_test, y_pred, zero_division=1))

    print("\n" + "-"*50 + "\n")

# Calculate and print the average accuracy for each model
print("Average Accuracy Across All Iterations:")
for name in cumulative_acc_score:
    average_accuracy = (cumulative_acc_score[name] / total_iterations) * 100
    print(f'Average Accuracy for {name}: {average_accuracy:.2f}%')
