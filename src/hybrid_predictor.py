import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from preprocess import preprocess

def hybrid_predict(test_file):
    # Load and preprocess test data
    X_test, y_test = preprocess(test_file)

    # Load trained classifier
    clf = joblib.load('../models/traffic_classifier.pkl')

    # Predict using Random Forest (supervised)
    y_pred = clf.predict(X_test)

    # Predict anomalies using Isolation Forest (unsupervised)
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    iso_model.fit(X_test)
    anomalies = iso_model.predict(X_test)
    anomalies = [1 if a == -1 else 0 for a in anomalies]

    # Combine results
    results = pd.DataFrame(X_test)
    results['Predicted_Label'] = y_pred
    results['Anomaly_Flag'] = anomalies

    # Save or display
    output_path = "../data/hybrid_results.csv"
    results.to_csv(output_path, index=False)
    print(f"✅ Hybrid results saved to {output_path}")

if __name__ == "__main__":
    hybrid_predict("../data/KDDTest_sample.csv")
