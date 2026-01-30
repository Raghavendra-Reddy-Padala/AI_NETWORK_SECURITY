import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import sys
import os

# Add src/ to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Try to import preprocess function
try:
    from preprocess import preprocess
    st.success("✅ preprocess.py loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load preprocess.py: {e}")

# Streamlit layout
st.set_page_config(page_title="Anomaly Detector", layout="centered")
st.title("📡 AI Network Anomaly Detector")

# Upload widget
uploaded_file = st.file_uploader("📁 Upload a network CSV file", type=["csv"])

# Main logic
if uploaded_file is not None:
    st.success("✅ File uploaded!")

    try:
        # Read raw data
        raw_df = pd.read_csv(uploaded_file, skiprows=1, header=None)

        # Preprocess features
        X, _ = preprocess(uploaded_file)

        # Run anomaly detection
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X)
        preds = model.predict(X)
        anomalies = [1 if p == -1 else 0 for p in preds]

        # Add anomaly column
        raw_df['Anomaly'] = anomalies

        # Show table
        st.subheader("🔍 Anomaly Detection Results")
        st.dataframe(raw_df)

        # Count & plot
        normal = anomalies.count(0)
        anomalous = anomalies.count(1)

        st.info(f"Normal: {normal} | Anomalies: {anomalous}")

        fig, ax = plt.subplots()
        ax.bar(['Normal', 'Anomaly'], [normal, anomalous], color=['green', 'red'])
        ax.set_title("Anomaly Distribution")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error during processing: {e}")
