import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

st.title("Batch Sales Prediction from Uploaded CSV")

# Prepare or load model (ندرب موديل ثابت هنا للبساطة)
@st.cache_data
def train_model():
    # Example training data (استبدلي ببياناتك الحقيقية لو عندك)
    data = {
        'TV': [230.1, 44.5, 17.2, 151.5, 180.8],
        'Radio': [37.8, 39.3, 45.9, 41.3, 10.8],
        'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4],
        'Sales': [22.1, 10.4, 9.3, 18.5, 12.9]
    }
    df = pd.DataFrame(data)
    df['TV_Radio'] = df['TV'] * df['Radio']
    df['TV_Newspaper'] = df['TV'] * df['Newspaper']
    df['Radio_Newspaper'] = df['Radio'] * df['Newspaper']
    df['TV_squared'] = df['TV'] ** 2
    df['Radio_squared'] = df['Radio'] ** 2
    df['Newspaper_squared'] = df['Newspaper'] ** 2
    df['log_TV'] = np.log1p(df['TV'])

    X = df[['TV', 'Radio', 'Newspaper', 'TV_Radio', 'TV_Newspaper',
            'Radio_Newspaper', 'TV_squared', 'Radio_squared', 'Newspaper_squared', 'log_TV']]
    y = df['Sales']

    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model()

uploaded_file = st.file_uploader("Upload your CSV file for prediction", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check required columns
    required_cols = ['TV', 'Radio', 'Newspaper']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Your CSV must contain the columns: {required_cols}")
    else:
        # Apply feature engineering on uploaded data
        df['TV_Radio'] = df['TV'] * df['Radio']
        df['TV_Newspaper'] = df['TV'] * df['Newspaper']
        df['Radio_Newspaper'] = df['Radio'] * df['Newspaper']
        df['TV_squared'] = df['TV'] ** 2
        df['Radio_squared'] = df['Radio'] ** 2
        df['Newspaper_squared'] = df['Newspaper'] ** 2
        df['log_TV'] = np.log1p(df['TV'])

        feature_cols = ['TV', 'Radio', 'Newspaper', 'TV_Radio', 'TV_Newspaper',
                        'Radio_Newspaper', 'TV_squared', 'Radio_squared', 'Newspaper_squared', 'log_TV']

        # Predict sales
        df['Predicted_Sales'] = model.predict(df[feature_cols])

        st.subheader("Predictions")
        st.dataframe(df)

        # Optionally let user download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='predicted_sales.csv',
            mime='text/csv',
        )
else:
    st.info("Upload a CSV file to get predictions.")
