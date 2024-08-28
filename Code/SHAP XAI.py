import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
print(1)
# Load your data (replace this with your actual data loading code)
df = pd.read_csv("Sleep_Stage_Combo4.csv")
drop_columns = ['SubNo', "Class", "SegNo", "Class2"]
# train test split
X = df.drop(drop_columns, axis=1)
y = df["Class2"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(2)
# Train a model (using Random Forest as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(3)
# Explain the model's predictions using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print(4)
# Get feature importance
feature_importance = np.abs(shap_values).mean(axis=0)

# Create a dataframe with feature names and importance values
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance
})
print(5)
# Sort features by importance
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

# Export feature importance to CSV
feature_importance_df.to_csv('shap_feature_importance.csv', index=False)

print("SHAP feature importance has been exported to 'shap_feature_importance.csv'")

# Optional: Plot SHAP summary
shap.summary_plot(shap_values, X_test, plot_type="bar")