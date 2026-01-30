import pandas as pd
import matplotlib.pyplot as plt

def visualize_anomalies(file_path):
    # Load anomaly results
    df = pd.read_csv(file_path)

    # Count normal vs anomaly
    counts = df['anomaly'].value_counts().sort_index()

    # Print counts
    print("Normal samples (0):", counts.get(0, 0))
    print("Anomalous samples (1):", counts.get(1, 0))

    # Plot bar chart
    labels = ['Normal', 'Anomaly']
    values = [counts.get(0, 0), counts.get(1, 0)]

    plt.bar(labels, values, color=['green', 'red'])
    plt.title("Network Traffic Classification")
    plt.xlabel("Traffic Type")
    plt.ylabel("Number of Flows")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_anomalies("../data/anomaly_output.csv")
