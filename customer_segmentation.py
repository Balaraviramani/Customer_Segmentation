# customer_segmentation.py - Script Version

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset from local file
df = pd.read_csv("ecommerce_customers.csv")

# Select relevant features for clustering
X = df[['Time_on_Website', 'Time_on_App', 'Length_of_Membership', 'Annual_Income']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_plot.png")
plt.show()

# Apply KMeans with chosen number of clusters (e.g., 4)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)
df['Cluster'] = kmeans.labels_

# Visualize clusters
sns.pairplot(df, hue='Cluster', vars=['Time_on_Website', 'Time_on_App', 'Length_of_Membership'])
plt.savefig("customer_clusters.png")
plt.show()

# Print cluster centers
print("\nCluster Centers (Standardized):")
print(kmeans.cluster_centers_)

# Save the clustered dataset
df.to_csv("segmented_customers.csv", index=False)
print("\nSegmented customer data saved to 'segmented_customers.csv'")
