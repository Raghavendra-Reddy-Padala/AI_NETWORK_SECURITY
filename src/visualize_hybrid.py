import pandas as pd
import matplotlib.pyplot as plt

# Load hybrid results
df = pd.read_csv("../data/hybrid_results.csv")

# Count labels
label_counts = df['Predicted_Label'].value_counts().sort_index()
anomaly_counts = df['Anomaly_Flag'].value_counts().sort_index()

# Print counts
print(f"🧠 Predicted Labels:\n{label_counts}")
print(f"🚨 Anomaly Flags:\n{anomaly_counts}")

# Plot side-by-side bar chart
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Classification
ax[0].bar(['Normal', 'Attack'], [label_counts.get(0, 0), label_counts.get(1, 0)], color=['green', 'red'])
ax[0].set_title("Classification Results")
ax[0].set_ylabel("Number of Flows")

# Anomalies
ax[1].bar(['Normal', 'Anomaly'], [anomaly_counts.get(0, 0), anomaly_counts.get(1, 0)], color=['blue', 'orange'])
ax[1].set_title("Anomaly Detection Results")

plt.tight_layout()
plt.show()
