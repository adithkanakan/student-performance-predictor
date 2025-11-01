import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("data/raw/student.csv", sep=';')

# Select only the relevant features
X = data[['G1', 'G2']]
y = data['G3']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, 'models/student_performance_model.pkl')

print("✅ Model trained and saved successfully!")
