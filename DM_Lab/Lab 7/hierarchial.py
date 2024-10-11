import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Step 1: Dataset Generation
def generate_dataset(n_clusters=3, points_per_cluster=50, cluster_std=0.5):
    np.random.seed(42)  # For reproducibility
    clusters = []
    for i in range(n_clusters):
        center = np.random.rand(2) * 10  # Random cluster center
        cluster_points = np.random.randn(points_per_cluster, 2) * cluster_std + center
        clusters.append(cluster_points)
    return np.vstack(clusters)

# Step 2: Distance Calculation
def compute_distance_matrix(data):
    return squareform(pdist(data))

def single_linkage(dist_matrix, clusters):
    min_dist = np.inf
    pair = (None, None)
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if dist_matrix[i, j] < min_dist:
                min_dist = dist_matrix[i, j]
                pair = (i, j)
    return pair

def complete_linkage(dist_matrix, clusters):
    max_dist = -np.inf
    pair = (None, None)
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if dist_matrix[i, j] > max_dist:
                max_dist = dist_matrix[i, j]
                pair = (i, j)
    return pair

# Step 3: Agglomerative Clustering
def agglomerative_clustering(data, linkage='single'):
    n_samples = data.shape[0]
    clusters = [[i] for i in range(n_samples)]  # Initialize clusters with indices
    dist_matrix = compute_distance_matrix(data)
    newPoints = []
    
    while len(clusters) > 1:
        if linkage == 'single':
            i, j = single_linkage(dist_matrix, clusters)
        elif linkage == 'complete':
            i, j = complete_linkage(dist_matrix, clusters)

        # Merge clusters
        clusters[i] += clusters[j]
        del clusters[j]
        data.append((data[i] + data[j] )/ 2)
        newPoints.append
        data.remove(data[i])
        data.remove(data[j])

        # Update distance matrix
        new_distances = []
        for k in range(len(clusters)):
            if k == i:
                continue
            if linkage == 'single':
                new_distance = np.min([dist_matrix[i, k], dist_matrix[j, k]])
            elif linkage == 'complete':
                new_distance = np.max([dist_matrix[i, k], dist_matrix[j, k]])
            new_distances.append(new_distance)

        dist_matrix = np.delete(dist_matrix, j, axis=0)  # Remove j row
        dist_matrix = np.delete(dist_matrix, j, axis=1)  # Remove j column
        dist_matrix = np.vstack([dist_matrix, np.concatenate([[0], new_distances])])
        dist_matrix = np.column_stack([dist_matrix, np.append(new_distances, np.inf)])

    return clusters[0]  # Return the final merged cluster indices

# Generate dataset
data = generate_dataset()

# Perform clustering
cluster_indices = agglomerative_clustering(data, linkage='single')

# Map cluster indices to data points
clustered_data_points = data[cluster_indices]

# Print the results
print("Clustered Data Points:")
print(clustered_data_points)

# Plotting the results
plt.scatter(data[:, 0], data[:, 1], c='blue', label='Data Points')
plt.scatter(clustered_data_points[:, 0], clustered_data_points[:, 1], c='red', label='Clustered Points')
plt.title('Agglomerative Hierarchical Clustering')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.legend()
plt.show()
