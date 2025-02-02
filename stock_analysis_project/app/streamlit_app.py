import streamlit as st
import joblib

# Load the best model
model = joblib.load('models/best_model.pkl')

st.title("Stock Price Prediction App")
ma_50 = st.number_input("Enter 50-day Moving Average")
ma_200 = st.number_input("Enter 200-day Moving Average")
daily_return = st.number_input("Enter Daily Return")

if st.button("Predict"):
    prediction = model.predict([[ma_50, ma_200, daily_return]])
    st.write(f"Predicted Close Price: ${prediction[0]:.2f}")
