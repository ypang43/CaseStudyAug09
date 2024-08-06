import pandas as pd

# Load the data
data_path = "Data.xlsx"
data = pd.read_excel(data_path, header=None)

# Display the first few rows of the data
print("Data Head:\n", data.head())
print("Data Shape:", data.shape)
