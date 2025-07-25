import streamlit as st
import pandas as pd
import joblib

# Load the trained model (no scaler involved)
model = joblib.load("selling_price_model.pkl")

# Set page configuration
st.set_page_config(page_title="Car Selling Price Predictor", layout="wide")
st.title("🚗 Car Selling Price Predictor")

# Sidebar for user inputs
st.sidebar.header("Enter Car Details")

# Input: Year of Purchase
year = st.sidebar.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2015)

# Input: Present Price of the car (in lakhs)
present_price = st.sidebar.number_input("Present Price (in Lakhs)", min_value=0.1, max_value=50.0, value=5.0)

# Input: Kilometers Driven
driven_kms = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000)

# Define Owner options and mapping to numeric values as per model training
owner_options = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']
owner_mapping = {
    'First Owner': 0,
    'Second Owner': 1,
    'Third Owner': 2,
    'Fourth & Above Owner': 3
}
owner = st.sidebar.selectbox("Owner Type", owner_options)

# Fuel Type options
fuel_options = ['Petrol', 'Diesel', 'CNG']
fuel = st.sidebar.selectbox("Fuel Type", fuel_options)

# Seller Type options
selling_type_options = ['Dealer', 'Individual']
selling_type = st.sidebar.selectbox("Seller Type", selling_type_options)

# Transmission options
transmission_options = ['Manual', 'Automatic']
transmission = st.sidebar.selectbox("Transmission", transmission_options)

# Prepare input data as per the model's expected features
input_data = {
    'Year': [year],  # Year of purchase
    'Present_Price': [present_price],  # Current price in lakhs
    'Driven_kms': [driven_kms],  # Total kilometers driven
    'Owner': [owner_mapping[owner]],  # Encoded owner type as integer
    # Fuel type encoding: only Diesel and Petrol exist in training features
    'Fuel_Type_Diesel': [1 if fuel == 'Diesel' else 0],
    'Fuel_Type_Petrol': [1 if fuel == 'Petrol' else 0],
    # If fuel is CNG, both above columns will be 0
    'Selling_type_Individual': [1 if selling_type == 'Individual' else 0],  # Binary encoding of seller type
    'Transmission_Manual': [1 if transmission == 'Manual' else 0]  # Binary encoding of transmission type
}

# Convert input dictionary to DataFrame
df_input = pd.DataFrame(input_data)

# Show the input data to the user
st.subheader("Your Input:")
st.write(df_input)

# When user clicks predict button
if st.button("Predict Price"):
    # Use the model to predict selling price
    prediction = model.predict(df_input)[0]
    # Show the prediction result
    st.success(f"💰 Estimated Selling Price: ₹ {round(prediction, 2)}")
