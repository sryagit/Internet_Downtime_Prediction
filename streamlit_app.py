import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image

# Load model
model = joblib.load('classifier.joblib')

# Display header image
image = Image.open('internet.png')
st.image(image.resize((1000, 300)))

# Prediction function
def internet_downtime_prediction(DownloadSpeed_Mbps, UploadSpeed_Mbps, Latency_ms, Jitter_ms, PacketLoss, Complaints):
    features = np.array([[DownloadSpeed_Mbps, UploadSpeed_Mbps, Latency_ms, Jitter_ms, PacketLoss, Complaints]])
    prediction = model.predict(features)
    return prediction[0]

# Main app
def main():
    st.title("🌐 Internet Downtime Prediction Web App")

    # City to Locality mapping
    city_localities = {
        "Mumbai": ["Andheri", "Bandra", "Dadar", "Colaba", "Kurla"],
        "Chennai": ["T Nagar", "Adyar", "Tambaram", "Velachery", "Anna Nagar"],
        "Delhi": ["Dwarka", "Rohini", "Saket", "Lajpat Nagar", "Karol Bagh"],
        "Bangalore": ["Indiranagar", "Koramangala", "Whitefield", "HSR Layout", "BTM"],
        "Pune": ["Baner", "Hinjewadi", "Kothrud", "Viman Nagar", "Wakad"]
    }

    # City selection
    City = st.selectbox("City", list(city_localities.keys()), index=None, placeholder="Select a City")

    # Locality selection (depends on City)
    if City:
        Locality = st.selectbox("Locality", city_localities[City], index=None, placeholder="Select a Locality")
    else:
        Locality = st.selectbox("Locality", [], index=None, placeholder="Select a City first")

    WeatherCondition = st.selectbox("Weather Condition", ['Sunny', 'Stormy', 'Cloudy', 'Rainy'], index=None, placeholder="Select Weather Condition")

    DownloadSpeed_Mbps = st.number_input(
        "Download Speed (Mbps) [Range: 0.0 – 100.0]",
        min_value=0.0,
        max_value=100.0,
        format="%.1f",   # Limits input to 1 decimal places
        step=0.1        # Increases/decreases by 0.1
    )

    UploadSpeed_Mbps = st.number_input(
        "Upload Speed (Mbps) [Range: 0.0 – 40.0]",
        min_value=0.0,
        max_value=40.0,
        format="%.1f",   # Limits input to 1 decimal places
        step=0.1        # Increases/decreases by 0.1
    )

    Latency_ms = st.number_input(
        "Latency (ms) [Range: 5.0 – 145.0]",
        min_value=5.0,
        max_value=145.0,
        format="%.1f",   # Limits input to 1 decimal places
        step=0.1        # Increases/decreases by 0.1
    )

    Jitter_ms = st.number_input(
        "Jitter (ms) [Range: 1.0 – 30.0]",
        min_value=1.0,
        max_value=30.0,
        format="%.1f",   # Limits input to 1 decimal places
        step=0.1        # Increases/decreases by 0.1
    )

    PacketLoss = st.number_input(
        "Packet Loss (%) [Range: 0.0 – 6.0]",
        min_value=0.0,
        max_value=6.0,
        format="%.1f",   # Limits input to 1 decimal places
        step=0.1        # Increases/decreases by 0.1
    )

    Complaints = st.number_input(
        "Complaints [Range: 0 – 100]",
        min_value=0,
        max_value=100,
        format="%d",   # Limits input to 0 decimal places
        step=1        # Increases/decreases by 1
    )

    if st.button("Predict Downtime"):
        result = internet_downtime_prediction(DownloadSpeed_Mbps, UploadSpeed_Mbps, Latency_ms, Jitter_ms, PacketLoss, Complaints)
        st.success(f"Predicted Downtime Category: {result}")

    if st.button("About"):
        st.text("Classifier: Support Vector Machine")
        st.text("Accuracy: 85.16%")
        st.text("Built by: Suraj R. Yadav")

if __name__ == '__main__':
    main()
