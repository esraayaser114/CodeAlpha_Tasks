import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load model and scaler
model = joblib.load('unemployment_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page config
st.set_page_config(page_title="Unemployment Predictor", layout="wide")
st.title("🔍 Unemployment Rate Predictor")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_unemployment_data.csv')

try:
    df = load_data()
except:
    st.error("⚠️ Make sure 'cleaned_unemployment_data.csv' exists in the directory.")
    st.stop()

# --- Data Summary ---
st.subheader(":bar_chart: Quick Data Overview:")

# Check if 'Area' column exists
if 'Area' in df.columns:
    area_avg = df.groupby('Area')[' Estimated Unemployment Rate (%)'].mean()
    fig, ax = plt.subplots()
    area_avg.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'], ax=ax)
    plt.title("Average Unemployment Rate by Area")
    plt.ylabel("Unemployment Rate (%)")
    st.pyplot(fig)

st.markdown(f"""
- ✅ **Overall Average Unemployment Rate**: `{df[' Estimated Unemployment Rate (%)'].mean():.2f}%`
- 🔺 **Highest Rate**: `{df[' Estimated Unemployment Rate (%)'].max():.2f}%`
- 🔻 **Lowest Rate**: `{df[' Estimated Unemployment Rate (%)'].min():.2f}%`
""")

# --- Prediction UI ---
st.subheader(":brain: Predict Unemployment Rate")
st.markdown("### ✏️ Enter your inputs:")

estimated_employed = st.number_input(":bar_chart: Estimated Employed (e.g. 5,000,000)", min_value=0.0, step=100000.0)
labour_participation = st.slider(":busts_in_silhouette: Labour Participation Rate (%)", 0.0, 100.0, 40.0)
region_encoded = st.number_input(":world_map: Region Code (e.g. 5)", min_value=0, step=1)
area = st.radio(":house: Area Type", ['Urban', 'unknown'])

year = st.selectbox(":calendar: Year", [2019, 2020, 2021])
month = st.slider(":date: Month", 1, 12)

area_urban = 1 if area == 'Urban' else 0
area_unknown = 1 if area == 'unknown' else 0

input_data = np.array([[estimated_employed, labour_participation, region_encoded, area_urban, area_unknown, year, month]])
input_scaled = input_data.copy()
input_scaled[:, [0, 1, 5, 6]] = scaler.transform(input_scaled[:, [0, 1, 5, 6]])

if st.button("🔮 Predict"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"📈 Predicted Unemployment Rate: {prediction:.2f}%")

# --- Model Evaluation ---
st.subheader(":pushpin: Model Evaluation")
st.markdown("""
- **MAE**: `~3.5`
- **MSE**: `~37.76`
- **RMSE**: `~6.14`
- **R² Score**: `~0.60`
""")
st.info("R² closer to 1 means better performance. You can improve the model with more data or feature engineering.")

# --- Interactive Dashboard ---
st.subheader(":bar_chart: Interactive Dashboard")

st.sidebar.title(":control_knobs: Filter Options")

selected_year = st.sidebar.selectbox("Select Year", df['Year'].unique())
selected_region = st.sidebar.selectbox("Select Region", df['Region'].unique())
area_filter = st.sidebar.radio("Area Type", ['All', 'Urban', 'Rural', 'unknown'])

filtered_df = df[df['Year'] == selected_year]
filtered_df = filtered_df[filtered_df['Region'] == selected_region]

if area_filter != 'All':
    filtered_df = filtered_df[filtered_df[f'Area_{area_filter}'] == 1]

st.markdown(f"### :mag: Unemployment Trend in {selected_region}, {selected_year}")
if not filtered_df.empty:
    line_data = filtered_df.sort_values(by='Month')
    fig2, ax2 = plt.subplots()
    ax2.plot(line_data['Month'], line_data[' Estimated Unemployment Rate (%)'], marker='o', color='orange')
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Unemployment Rate (%)")
    ax2.set_title("Monthly Unemployment Rate")
    st.pyplot(fig2)
else:
    st.warning("⚠️ No data matches the selected filters.")
