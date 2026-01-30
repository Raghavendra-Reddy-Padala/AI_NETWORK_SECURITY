import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import sys
import os

# ----------------------------------------------------
# ✅ MUST BE THE FIRST STREAMLIT COMMAND
# ----------------------------------------------------
st.set_page_config(page_title="Anomaly Detector", layout="centered")

# ----------------------------------------------------
# Add src/ to Python path
# ----------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# ----------------------------------------------------
# Import preprocess function
# ----------------------------------------------------
try:
    from preprocess import preprocess
    st.success("✅ preprocess.py loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load preprocess.py: {e}")
    st.stop()   # stop app if preprocess fails

# ----------------------------------------------------
# UI
# ----------------------------------------------------
st.title("📡 AI Network Anomaly Detector")
st.write("Upload a network traffic CSV file to detect anomalies.")

uploaded_file = st.file_uploader(
    "📁 Upload a network CSV file",
    type=["csv"]
)

# ----------------------------------------------------
# Main Logic
# ----------------------------------------------------
if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    try:
        # Read raw data
        raw_df = pd.read_csv(uploaded_file, skiprows=1, header=None)

        # 🔴 IMPORTANT: reset file pointer before reuse
        uploaded_file.seek(0)

        # Preprocess features
        X, _ = preprocess(uploaded_file)

        # Run anomaly detection
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X)
        preds = model.predict(X)

        # Convert predictions to anomaly flag
        anomalies = [1 if p == -1 else 0 for p in preds]

        # Add anomaly column
        raw_df["Anomaly"] = anomalies

        # ------------------------------------------------
        # Results Table
        # ------------------------------------------------
        st.subheader("🔍 Anomaly Detection Results")
        st.dataframe(raw_df, use_container_width=True)

        # ------------------------------------------------
        # Metrics
        # ------------------------------------------------
        normal = anomalies.count(0)
        anomalous = anomalies.count(1)

        col1, col2 = st.columns(2)
        col1.metric("Normal Records", normal)
        col2.metric("Anomalies Detected", anomalous)

        # ------------------------------------------------
        # Visualization
        # ------------------------------------------------
        fig, ax = plt.subplots()
        ax.bar(
            ["Normal", "Anomaly"],
            [normal, anomalous],
            color=["green", "red"]
        )
        ax.set_title("Anomaly Distribution")
        ax.set_ylabel("Count")

        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error during processing: {e}")
