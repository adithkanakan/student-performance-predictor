import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model
model = joblib.load("models/student_performance_model.pkl")

st.title("🎓 Student Performance Predictor")
st.write("Predict a student's final grade (G3) based on their study habits and background.")

# Input fields
age = st.number_input("Age", min_value=15, max_value=22, value=17)
studytime = st.slider("Study Time (1 = <2h, 2 = 2–5h, 3 = 5–10h, 4 = >10h)", 1, 4, 2)
failures = st.slider("Number of past class failures", 0, 4, 0)
absences = st.slider("Number of absences", 0, 93, 5)
G1 = st.number_input("1st Period Grade (G1)", 0, 20, 10)
G2 = st.number_input("2nd Period Grade (G2)", 0, 20, 10)

# Collect inputs in correct order (the same as model training)
input_data = pd.DataFrame({
    'G1': [G1],
    'G2': [G2]
})

# Predict button
if st.button("Predict Final Grade (G3)"):
    prediction = model.predict(input_data)
    st.success(f"🎯 Predicted Final Grade (G3): {prediction[0]:.2f}")
