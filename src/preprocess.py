import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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

    # Read CSV, skip header row if present
    df = pd.read_csv(file_path, names=col_names, skiprows=1)

    # Encode categorical columns to numeric
    for col in ['protocol_type', 'service', 'flag']:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Convert labels: normal -> 0, attack -> 1
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    # Separate features and labels
    y = df['label']
    X = df.drop('label', axis=1)

    # Ensure all features are numeric and convert to float
    X = X.apply(pd.to_numeric, errors='coerce')

    # Check for any NaNs after conversion (optional)
    if X.isnull().values.any():
        print("Warning: NaNs found in features after conversion. Filling with 0.")
        X = X.fillna(0)

    # Scale features to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

if __name__ == "__main__":
    X, y = preprocess('../data/KDDTrain_sample.csv')
    print(f"Features shape: {X.shape}")
    print("Labels sample:")
    print(y.head())
