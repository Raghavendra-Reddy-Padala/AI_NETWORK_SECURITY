import pandas as pd
from sklearn.ensemble import IsolationForest
from preprocess import preprocess

def run_anomaly_detection(file_path):
    # Load and preprocess features (X) only
    X, _ = preprocess(file_path)

    # Train Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)

    # Predict (-1 = anomaly, 1 = normal)
    predictions = model.predict(X)
    anomaly_flags = [1 if p == -1 else 0 for p in predictions]

    # Load original raw data for reference
    original_df = pd.read_csv(file_path, skiprows=1, header=None)



    # Add anomaly flag to original data
    original_df['anomaly'] = anomaly_flags

    # Save to new CSV
    output_path = "../data/anomaly_output.csv"
    original_df.to_csv(output_path, index=False)

    print(f"🚨 Anomalies Detected: {sum(anomaly_flags)} out of {len(anomaly_flags)} samples")
    print(f"✅ Anomaly results saved to {output_path}")

if __name__ == "__main__":
    run_anomaly_detection("../data/KDDTest_sample.csv")
