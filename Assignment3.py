import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

data = pd.read_csv("Grouped.csv", index_col=0)
ndata = pd.read_csv("Grouped-Normalised.csv", index_col=0)

data_T = pd.read_csv("Grouped.csv", index_col=0).T
ndata_T = pd.read_csv("Grouped-Normalised.csv", index_col=0).T
print(ndata_T["GDP per capita"])


# %%
# perform KMeans clustering
def perform_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    cluster_centers = kmeans.cluster_centers_
    return clusters, cluster_centers
# %%


# Cluster for High Development , Low emmissions.
dev_threshold = 0.18  # Adjust as needed
emissions_threshold = 0.07  # Adjust as needed
high_dev_low_emissions = ndata_T[(ndata_T["GDP per capita"] > dev_threshold) & (
    ndata_T["CO2 emissions (kt)"] < emissions_threshold)]

num_clusters = 3
clusters, cluster_centers = perform_clustering(
    high_dev_low_emissions, num_clusters)

plt.figure(figsize=(10, 6))
plt.scatter(high_dev_low_emissions["GDP per capita"],
            high_dev_low_emissions["CO2 emissions (kt)"], c=clusters, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            marker='x', s=100, color='red')  # Plot cluster centers
plt.title("High Development, Low Emissions Countries")
plt.xlabel("GDP per capita")
plt.ylabel("CO2 emissions (kt)")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

