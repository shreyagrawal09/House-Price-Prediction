import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model and features
model = joblib.load("linear_model.pkl")
features = joblib.load("feature_names.pkl")

st.title("🏠 California House Price Prediction App")

st.write("""
### Enter the details of the house:
""")

# Create input fields for all features
input_data = []
for feature in features:
    val = st.number_input(f"{feature}", step=0.1)
    input_data.append(val)

# When user clicks Predict button
if st.button("Predict Price"):
    # Convert input to numpy array and reshape
    input_array = np.array(input_data).reshape(1, -1)

    # Predict using the model
    prediction = model.predict(input_array)[0]

    st.success(f"💰 Estimated House Price: ${prediction * 100000:,.2f}")
