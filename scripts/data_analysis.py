import pandas as pd
import os

# Define the file path
file_path = os.path.join("data", "raw", "student.csv")

# Read the CSV correctly (semicolon separated)
data = pd.read_csv(file_path, sep=';')

# Display first few rows
print("✅ Preview of data:")
print(data.head())

# Display dataset info
print("\n📊 Dataset Info:")
print(data.info())

# Display basic statistics
print("\n📈 Basic Statistics:")
print(data.describe())

# Check correlation between grade columns
print("\n📘 Correlation between scores (G1, G2, G3):")
print(data[['G1', 'G2', 'G3']].corr())
