import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image

# Load model
model = joblib.load('Random_Forest_model.joblib')

# Display header image
image = Image.open('internet.png')
st.image(image.resize((1000, 300)))

def internet_downtime_prediction(City, Locality, WeatherCondition, DownloadSpeed_Mbps, UploadSpeed_Mbps, Latency_ms, Jitter_ms, PacketLoss, Complaints):
    features = pd.DataFrame([{
        'City': City,
        'Locality': Locality,
        'WeatherCondition': WeatherCondition,
        'DownloadSpeed_Mbps': DownloadSpeed_Mbps,
        'UploadSpeed_Mbps': UploadSpeed_Mbps,
        'Latency_ms': Latency_ms,
        'Jitter_ms': Jitter_ms,
        'PacketLoss_%': PacketLoss,   
        'Complaints': Complaints
    }])
    prediction = model.predict(features)
    return prediction[0]

# Main app
def main():
    st.title("🌐 Internet Downtime Prediction")

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
        "Download Speed (Mbps) [Range: 0.0 – 200]",
        min_value=0.0,
        max_value=200.0,
        format="%.1f",   # Limits input to 1 decimal places
        step=0.1        # Increases/decreases by 0.1
    )

    UploadSpeed_Mbps = st.number_input(
        "Upload Speed (Mbps) [Range: 0.0 – 50.0]",
        min_value=0.0,
        max_value=50.0,
        format="%.1f",   # Limits input to 1 decimal places
        step=0.1        # Increases/decreases by 0.1
    )

    Latency_ms = st.number_input(
        "Latency (ms) [Range: 0.5 – 145.0]",
        min_value=0.5,
        max_value=145.0,
        format="%.1f",   # Limits input to 1 decimal places
        step=0.1        # Increases/decreases by 0.1
    )

    Jitter_ms = st.number_input(
        "Jitter (ms) [Range: 0.1 – 35.0]",
        min_value=0.1,
        max_value=35.0,
        format="%.1f",   # Limits input to 1 decimal places
        step=0.1        # Increases/decreases by 0.1
    )

    PacketLoss = st.number_input(
        "Packet Loss (%) [Range: 0.0 – 7.0]",
        min_value=0.0,
        max_value=7.0,
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
        result = internet_downtime_prediction(
            City, Locality, WeatherCondition, 
            DownloadSpeed_Mbps, UploadSpeed_Mbps, 
            Latency_ms, Jitter_ms, PacketLoss, Complaints
        )
    
        # Default color for label
        label_html = f'<span style="color: red; font-weight:bold;">Predicted Downtime Category:</span> '
    
        # Set color based on prediction
        if result == "Low_Downtime":
            value_html = f'<span style="color: lightgreen; font-weight:bold;">{result}</span>'
        elif result == "Moderate_Downtime":
            value_html = f'<span style="color: yellow; font-weight:bold;">{result}</span>'
        elif result == "High_Downtime":
            value_html = f'<span style="color: orange; font-weight:bold;">{result}</span>'
        else:
            value_html = f'<span style="color: black;">{result}</span>'
    
        # Display colored prediction
        st.markdown(label_html + value_html, unsafe_allow_html=True)
    
        # Show About info after prediction
        st.info(
            """
            **Classifier:** Random Forest Classifier  
            **Accuracy:** 92.00 %  
            **Built by:** Suraj R. Yadav
            """
        )


if __name__ == '__main__':
    main()
