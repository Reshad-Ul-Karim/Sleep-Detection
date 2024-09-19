import pandas as pd

# Load the dataset
df = pd.read_csv("Sleep_Stage_Combo2.csv")

# Check unique classes in the "Class2" column
unique_classes = df["Class2"].unique()
print(f"Unique classes in the data: {unique_classes}")
