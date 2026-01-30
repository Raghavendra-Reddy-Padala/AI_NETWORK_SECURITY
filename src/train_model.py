import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

def preprocess(file_path):
    col_names = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
        "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
        "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
        "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
    ]

    df = pd.read_csv(file_path, names=col_names, skiprows=1)
    for col in ['protocol_type', 'service', 'flag']:
        df[col] = LabelEncoder().fit_transform(df[col])

    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    y = df['label']
    X = df.drop('label', axis=1)
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

if __name__ == "__main__":
    # Load and preprocess data
    X, y = preprocess('../data/KDDTrain_sample.csv')

    # Split dataset into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train model
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Print evaluation metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # ✅ Save the trained model
    import joblib
    joblib.dump(clf, '../models/traffic_classifier.pkl')
    print("✅ Model saved to ../models/traffic_classifier.pkl")
