import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Features used for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Train model
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(X)

# Save model
joblib.dump(kmeans, "customer_segmentation_model.pkl")

print("Model Trained Successfully")