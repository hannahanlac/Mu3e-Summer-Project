import awkward as ak
import awkward0
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import os
import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import cKDTree

def load_data(file_path):
    table = pq.read_table(file_path)  # Load Parquet file as an Arrow table
    print(table)
    awk_array = ak.from_arrow(table)       # Convert Arrow table to an Awkward Array
    print(awk_array)

    # Debugging: Check the first few rows
    print(f"Loaded dataset: {len(awk_array['hit_ID'])} hits")

    print("First few frame values:", ak.to_numpy(awk_array['frame_array'])[:10])
    print("Frames from index 5000-5020:", ak.to_numpy(awk_array['frame_array'])[5000:5020])

    print("First few hit IDs:", ak.to_numpy(awk_array['hit_ID'])[:10])
    print("First few pixelx values", ak.to_numpy(awk_array['pixelx_array'])[:10])
    
    """frame_values = ak.to_numpy(awk_array['frame_array'])  # Convert to NumPy
    print("Are frames sorted?", np.all(frame_values[:-1] <= frame_values[1:]))  # Check if sorted"""

    return awk_array


def batch_data(awk_array, frames_per_batch=1):
    unique_frames = np.unique(ak.to_numpy(awk_array['frame_array']))  # Get unique frame IDs
    batches = []

    for i in range(0, len(unique_frames), frames_per_batch):
        selected_frames = unique_frames[i:i + frames_per_batch]  # Select a chunk of frames
        mask = np.isin(ak.to_numpy(awk_array['frame_array']), selected_frames).flatten()  
        batch = awk_array[mask] 

        # Print batch summary
        print(f"Batch {len(batches) + 1}: {len(selected_frames)} frames, {len(batch['hit_ID'])} hits")
        print(f"Frames in batch: {selected_frames}")
        print(batch)
        print()

        batches.append(batch)

    print(batches[0][72]['layer_array']) 
    # prints the value associated layer_array for hit indexed number 72 in frame 0 / batch 0

    return batches

def build_graph(batch, k_neighbors=5):
    """
    Constructs a graph for a batch based on spatial proximity (KNN).
    
    Args:
        batch: Awkward array containing hits for one batch.
        k_neighbors: Number of nearest neighbors to connect.

    Returns:
        G: A NetworkX graph representing the batch.
    """

    # Convert Awkward arrays to NumPy
    gx = ak.to_numpy(batch['gx'])
    gy = ak.to_numpy(batch['gy'])
    gz = ak.to_numpy(batch['gz'])

    # Ensure arrays are not empty
    if len(gx) == 0 or len(gy) == 0 or len(gz) == 0:
        print("Warning: Empty batch detected!")
        return nx.Graph()  # Return an empty graph

    # Stack into coordinate array
    coords = np.column_stack((gx, gy, gz))  # Alternative to vstack.T, shape (N, 3)

    # Debugging: Print the shape of the coordinates
    print(f"Batch size: {len(batch['hit_ID'])}, Coords shape: {coords.shape}")

    # Ensure coords has shape (N, 3)
    if coords.shape[1] != 3:
        raise ValueError(f"Unexpected coordinate shape: {coords.shape}")

    G = nx.Graph()  # Initialize an empty graph

    # Add nodes
    for i, (x, y, z) in enumerate(coords):
        G.add_node(i, gx=x, gy=y, gz=z, hit_ID=batch['hit_ID'][i], tid=batch['tid_array'][i])

    # Find nearest neighbors for each hit
    tree = cKDTree(coords)  # KDTree for fast nearest neighbor search
    for i, coord in enumerate(coords):
        _, indices = tree.query(coord, k=k_neighbors + 1)  # +1 to exclude self
        for j in indices[1:]:  # Skip self (first index)
            G.add_edge(i, j)  # Add an edge between neighbors

    return G



file_path = 'ProcessedData/signal1_96_32652/train_data/train_data.parquet'
awk_data = load_data(file_path)
batches = batch_data(awk_data)
batch_graphs = [build_graph(batch) for batch in batches]

print(f"Generated {len(batch_graphs)} graphs!")  
print("First batch graph details:", batch_graphs[0])  # Print first graph


"""class EdgeClassifier(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.dense1 = Dense(hidden_dim, activation='relu')
        self.batch_norm = BatchNormalization()
        self.dense2 = Dense(hidden_dim, activation='relu')
        self.output_layer = Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.batch_norm(x)
        x = self.dense2(x)
        return self.output_layer(x)

def train_model(x_train, y_train, model, epochs=1, batch_size=32):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

def cluster_hits(features, predictions, threshold=0.8):

    mask = (predictions > threshold).squeeze()
    print(f"Number of hits passing threshold: {np.sum(mask)}")

    if np.sum(mask) == 0:  # No hits above the threshold
        print("No valid hits for clustering in this batch.")
        return np.full(len(features), -1)  # Assign all hits as noise (-1)

    # Ensure enough points for DBSCAN to work
    if len(features[mask]) < 2:
        print("Stopping: Last batch had too few hits for clustering.")
        return np.full(len(features), -1)  # Assign noise

    print(f"Clustering {len(features[mask])} hits...")
    clustering = DBSCAN(eps=1.5, min_samples=2).fit(features[mask])
    return clustering.labels_


features = np.vstack([stack3Dcoords(batch) for batch in batches])
labels = np.concatenate([np.ones(len(batch)) for batch in batches])  # Placeholder labels
indices = np.arange(len(features))
np.random.shuffle(indices)
features, labels = features[indices], labels[indices]


model = EdgeClassifier(input_dim=3, hidden_dim=16)
train_model(features, labels, model)

model.save('track_finding_gnn')

predictions = model.predict(features)
all_clusters = []
all_hit_data = []
for i, batch in enumerate(batches):
    batch_features = stack3Dcoords(batch)
    batch_predictions = model.predict(batch_features)
    
    # Perform clustering on the current batch
    batch_clusters = cluster_hits(batch_features, batch_predictions)

    # Store hit information
    for j, hit_id in enumerate(ak.to_numpy(batch['hit_ID'])):
        all_hit_data.append({
            "cluster_ID": batch_clusters[j] if batch_clusters[j] != -1 else None,  # Assign cluster ID, None for noise
            "hit_ID": hit_id,
            "frame_array": ak.to_numpy(batch['frame_array'])[j],
            "pixelx_array": ak.to_numpy(batch['pixelx_array'])[j],
            "pixely_array": ak.to_numpy(batch['pixely_array'])[j],
            "layer_array": ak.to_numpy(batch['layer_array'])[j],
            "station_array": ak.to_numpy(batch['station_array'])[j],
            "ladder_array": ak.to_numpy(batch['ladder_array'])[j],
            "chip_array": ak.to_numpy(batch['chip_array'])[j],
            "tid_array": ak.to_numpy(batch['tid_array'])[j],
            "gx": ak.to_numpy(batch['gx'])[j],
            "gy": ak.to_numpy(batch['gy'])[j],
            "gz": ak.to_numpy(batch['gz'])[j]
        })
    
    all_clusters.append(batch_clusters)


# Convert list to DataFrame
df = pd.DataFrame(all_hit_data)

# Remove noise hits (DBSCAN labels noise as -1)
df = df.dropna(subset=["cluster_ID"]).astype({"cluster_ID": int}) 

# Save to CSV
output_csv = "ProcessedData/signal1_96_32652/GNNoutput/tracks_output.csv"
df.to_csv(output_csv, index=False)

print(f"CSV file saved: {output_csv}")

plt.scatter(features[:, 0], features[:, 1], c=all_clusters, cmap='viridis')
plt.xlabel('gx')
plt.ylabel('gy')
plt.title('Hit Clustering')
plt.colorbar()
plt.show()"""
