import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define file paths
raw_data_path = os.path.join("data", "raw", "student.csv")
processed_data_path = os.path.join("data", "processed", "student_processed.csv")

# Load the dataset
data = pd.read_csv(raw_data_path, sep=';')

# Encode categorical columns
label_encoder = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = label_encoder.fit_transform(data[col])

# Split into features (X) and target (y)
X = data.drop(columns=['G3'])
y = data['G3']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed dataset
os.makedirs("data/processed", exist_ok=True)
processed_data = pd.concat([X_train, y_train], axis=1)
processed_data.to_csv(processed_data_path, index=False)

print("✅ Data preprocessing complete.")
print(f"Processed data saved to: {processed_data_path}")
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
