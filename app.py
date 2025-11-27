import streamlit as st
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = load('ridge_model_.pkl')

# Page configuration
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("California House Price Prediction Web App")
st.write("Predict house prices based on California housing dataset features and visualize locations.")

# --- Sidebar for user inputs ---
st.sidebar.header("Input Features")

MedInc = st.sidebar.slider("Median Income (MedInc)", 0.0, 20.0, 5.0)
HouseAge = st.sidebar.slider("House Age (HouseAge)", 0.0, 100.0, 20.0)
AveRooms = st.sidebar.slider("Average Rooms (AveRooms)", 1.0, 20.0, 5.0)
AveBedrms = st.sidebar.slider("Average Bedrooms (AveBedrms)", 1.0, 10.0, 2.0)
Population = st.sidebar.slider("Population", 1.0, 5000.0, 1000.0)
AveOccup = st.sidebar.slider("Average Occupancy (AveOccup)", 1.0, 10.0, 3.0)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.77)
Longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -122.42)

input_data = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]],
                          columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])

# --- Main layout: two columns ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Summary")
    st.dataframe(input_data)

with col2:
    # Predict button
    if st.button("Predict House Price"):
        price = model.predict(input_data)[0]
        st.success(f"Predicted House Price: ${price:,.2f}")

        # Map
        st.subheader("House Location on Map")
        map_data = pd.DataFrame({'lat': [Latitude], 'lon': [Longitude]})
        st.map(map_data)

        # Feature importance chart
        st.subheader("Feature Importance")
        coef = model.coef_
        features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        fig, ax = plt.subplots()
        sns.barplot(x=features, y=coef, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# --- Batch prediction ---
st.subheader("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV with 8 features", type="csv")

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    
    # Check columns
    required_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    if not all(col in batch_data.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        batch_preds = model.predict(batch_data[required_cols])
        batch_data['PredictedPrice'] = batch_preds
        st.dataframe(batch_data)

        # Download button
        csv = batch_data.to_csv(index=False)
        st.download_button("Download Predictions", csv, "predictions.csv")

        # Map all houses
        st.subheader("Map of Predicted Houses")
        st.map(batch_data[['Latitude', 'Longitude']])