import joblib
import pandas as pd
from preprocess import preprocess
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def run_prediction(test_file):
    print("🔄 Loading model...")
    model = joblib.load('../models/traffic_classifier.pkl')

    print("📦 Preprocessing test data...")
    X_test, y_test = preprocess(test_file)

    print("🤖 Running predictions...")
    predictions = model.predict(X_test)

    print("\n📊 Evaluation Results:")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))

if __name__ == "__main__":
    run_prediction("../data/KDDTest_sample.csv")
