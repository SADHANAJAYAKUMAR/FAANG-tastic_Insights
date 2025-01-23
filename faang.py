import streamlit as st
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.pyfunc
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
@st.cache_resource
def load_model_and_scaler():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_name = "Random Forest Regressor"
    model_version = "6"
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    
    std_scaler = pickle.load(open('F:\PROJECT\MINI PROJ 4\standard_scaler.pkl', 'rb'))
    
    return model, std_scaler

# Load model and scaler once (cached)
model, std_scaler = load_model_and_scaler()

# Load Label Encoders for categorical features
debt_to_equity_label_encoder = pickle.load(open("F:\PROJECT\MINI PROJ 4\debt_to_equity_label_encoder.pkl", 'rb'))
eps_label_encoder = pickle.load(open("F:\PROJECT\MINI PROJ 4\eps_label_encoder.pkl", 'rb'))
pe_ratio_label_encoder = pickle.load(open("F:\PROJECT\MINI PROJ 4\pe_ratio_label_encoder.pkl", 'rb'))
price_to_book_ratio_label_encoder = pickle.load(open("F:\PROJECT\MINI PROJ 4\price_to_book_ratio_label_encoder.pkl", 'rb'))

# Streamlit user interface
st.title("FAANG - Stock Closing Price Predictor")
st.markdown("""
This Streamlit application predicts the closing price of FAANG stocks based on the provided input data.
""")

st.header("Input Stock Details")

# Input fields for user to enter stock data
company_name = st.selectbox("Select Company Stock", ["AMAZON", "APPLE", "FACEBOOK", "GOOGL", "NETFLIX"])

# Stock encoding (one-hot encoding for companies)
Ticker_AMZN = 1 if company_name == "AMAZON" else 0
Ticker_AAPL = 1 if company_name == "APPLE" else 0
Ticker_META = 1 if company_name == "FACEBOOK" else 0
Ticker_GOOGL = 1 if company_name == "GOOGL" else 0
Ticker_NFLX = 1 if company_name == "NETFLIX" else 0

# Input numerical fields
open_price = st.number_input("Open Price", value=100.0)
high_price = st.number_input("High Price", value=105.0)
low_price = st.number_input("Low Price", value=95.0)
volume = st.number_input("Volume", value=1_000_000.0)

# Day of the week selection
day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
dow_encoded = {
    "Monday": [1, 0, 0, 0, 0],
    "Tuesday": [0, 1, 0, 0, 0],
    "Wednesday": [0, 0, 1, 0, 0],
    "Thursday": [0, 0, 0, 1, 0],
    "Friday": [0, 0, 0, 0, 1],
}[day_of_week]

# Month selection
month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
month_encoded = {
    "January": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "February": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "March": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "April": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "May": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "June": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "July": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "August": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "September": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "October": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "November": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "December": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}[month]

# Year input
year_encoded = st.number_input("Year", value=2024)

# Default company values
company_values = {
    "AMAZON": (2000000000000.0, 45.496414, 4.18, 66.756, 8.4372225),
    "APPLE": (2845000000000.0, 35.789955, 6.57, 139.4925, 23.0003087),
    "FACEBOOK": (1470000000000.0, 29.612986, 19.56, 24.235, 9.359326),
    "GOOGL": (2020000000000.0, 23.492826, 6.97, 9.549, 6.7086606),
    "NETFLIX": (645000000000.0, 42.8245, 17.67, 70.338, 14.262457)
}

market_cap, pe_ratio, eps, debt_to_equity, price_to_book_ratio = company_values.get(company_name, (0, 0, 0, 0, 0))

# Now create the numerical feature list
numerical_features = [open_price, high_price, low_price, volume, market_cap, pe_ratio, eps, debt_to_equity, price_to_book_ratio]

# Create the categorical feature list (already one-hot encoded)
categorical_features = [Ticker_AMZN, Ticker_AAPL, Ticker_META, Ticker_GOOGL, Ticker_NFLX, *dow_encoded, *month_encoded, year_encoded]

# Combine the features (numerical + categorical)
all_features = numerical_features + categorical_features  # This will have 32 features in total

# Scaling all features (numerical + categorical)
scaled_features = std_scaler.transform([all_features])[0]

# Make prediction when button is clicked
if st.button("Predict the Closing Price"):
    with st.spinner('Predicting...'):
        predicted_closing_price = model.predict([scaled_features])[0]
    
    st.header("Predicted Closing Price")
    st.subheader(f"${predicted_closing_price:.4f}")
