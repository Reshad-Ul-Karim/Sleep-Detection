import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import uniform

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

# Initialize the SVM pipeline
svm_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(class_weight='balanced', random_state=160))
])

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'svm__C': uniform(1, 100),  # Range from 1 to 101
    'svm__gamma': uniform(0.001, 0.1),  # Range for gamma
    'svm__degree': [2, 3, 4],  # Only for 'poly' kernel
    'svm__kernel': ['linear', 'rbf', 'poly'],  # Try different kernels
    'svm__coef0': uniform(0, 1),  # Only for 'poly' kernel
}

# Subset the data using the top 34 features
X = df[top_34_features]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=160, stratify=y)

# Perform RandomizedSearchCV for hyperparameter optimization
random_search = RandomizedSearchCV(svm_model, param_distributions=param_dist, n_iter=20, scoring='accuracy', n_jobs=-1, cv=3, random_state=160)
random_search.fit(X_train, y_train)

# Get the best parameters and accuracy
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Make predictions with the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Get classification report
class_report = classification_report(y_test, y_pred)

# Print the result
print(f"Top 34 features accuracy: {accuracy:.4f}")
print(f"Best parameters: {best_params}\n")
print(f"Classification Report:\n{class_report}")

# Save the results to a text file
with open("SVM_Hyperparameter_Results_Top34.txt", "w") as output_file:
    output_file.write(f"Top 34 features:\n")
    output_file.write(f"Best parameters: {best_params}\n")
    output_file.write(f"Accuracy: {accuracy:.4f}\n")
    output_file.write(f"Classification Report:\n{class_report}\n")
    output_file.write("="*80 + "\n")
