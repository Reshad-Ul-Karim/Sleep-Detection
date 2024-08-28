import pandas as pd

# Load your dataset
df = pd.read_csv("E:\Paper\Brain signal\Sleep_Stage_Combo2.csv")

# Check which columns have all rows with the same value
uniform_columns = df.columns[df.nunique() == 1]

# Display the columns with uniform values
print("Columns where all rows have the same value:")
print(uniform_columns)

