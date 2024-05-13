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

# %%
# Cluster for Low Development , High emmissions.
# Filter countries based on criteria
low_dev_high_emissions = ndata_T[(ndata_T["GDP per capita"] < dev_threshold) & (
    ndata_T["CO2 emissions (kt)"] > emissions_threshold)]
num_clusters = 3
clusters, cluster_centers = perform_clustering(
    low_dev_high_emissions, num_clusters)

plt.figure(figsize=(10, 6))
plt.scatter(low_dev_high_emissions["GDP per capita"],
            low_dev_high_emissions["CO2 emissions (kt)"], c=clusters, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            marker='x', s=100, color='red')  # Plot cluster centers
plt.title("Low Development, High Emissions Countries")
plt.xlabel("GDP per capita")
plt.ylabel("CO2 emissions (kt)")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
# %%
# Cluster for Low Forest Area , High emmissions.
frst_threshold = 0.025
# Filter countries based on criteria
low_frst_high_emissions = ndata_T[(ndata_T["Forest area in KM"] < frst_threshold) & (
    ndata_T["CO2 emissions (kt)"] > emissions_threshold)]
num_clusters = 3
clusters, cluster_centers = perform_clustering(
    low_frst_high_emissions, num_clusters)


plt.figure(figsize=(10, 6))
plt.scatter(low_frst_high_emissions["Forest area in KM"],
            low_frst_high_emissions["CO2 emissions (kt)"], c=clusters, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            marker='x', s=100, color='red')  # Plot cluster centers
plt.title("Low Forest Area, High Emissions Countries")
plt.xlabel("Forest Area In KM ")
plt.ylabel("CO2 emissions (kt)")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# %%
# Cluster for High access to electricity , Low emmissions.
# Filter countries based on criteria
elec_threshold = 0.025
high_elec_low_emissions = ndata_T[(ndata_T["Access to electricity %"] > elec_threshold) & (
    ndata_T["CO2 emissions (kt)"] < emissions_threshold)]

num_clusters = 3
clusters, cluster_centers = perform_clustering(
    high_elec_low_emissions, num_clusters)

plt.figure(figsize=(10, 6))
plt.scatter(high_elec_low_emissions["Access to electricity %"],
            high_elec_low_emissions["CO2 emissions (kt)"], c=clusters, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            marker='x', s=100, color='red')  # Plot cluster centers
plt.title("High Access to Electricity,  Low Emissions Countries")
plt.xlabel("Access To Electricity")
plt.ylabel("CO2 emissions (kt)")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# %%
x_data = ndata_T["Forest area in KM"]
y_data = ndata_T["CO2 emissions (kt)"]


def model_function(x, a, b):
    return a * x + b


popt, pcov = curve_fit(model_function, x_data, y_data)
y_fit = model_function(x_data, *popt)

plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, y_fit, color='red', label='Linear Fit')
plt.xlabel('Forest area in KM')
plt.ylabel('CO2 emissions (kt)')
plt.title('Linear Fit: Forest area vs CO2 emissions')
plt.legend()
plt.grid(True)
plt.show()

# %%
x_data = ndata_T["GDP per capita"]
y_data = ndata_T["CO2 emissions (kt)"]


def model_function(x, a, b):
    return a * x + b


popt, pcov = curve_fit(model_function, x_data, y_data)
y_fit = model_function(x_data, *popt)

plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, y_fit, color='red', label='Linear Fit')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 emissions (kt)')
plt.title('Linear Fit: GDP per capita vs CO2 emissions')
plt.legend()
plt.grid(True)
plt.show()
x_data = ndata_T["GDP per capita"]
y_data = ndata_T["CO2 emissions (kt)"]

# %%
x_data = ndata_T["Access to electricity %"]
y_data = ndata_T["CO2 emissions (kt)"]


def model_function(x, a, b):
    return a * x + b


popt, pcov = curve_fit(model_function, x_data, y_data)
y_fit = model_function(x_data, *popt)

plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, y_fit, color='red', label='Linear Fit')
plt.xlabel('Access to electricity %')
plt.ylabel('CO2 emissions (kt)')
plt.title('Linear Fit: Access to electricity % vs CO2 emissions')
plt.legend()
plt.grid(True)
plt.show()
