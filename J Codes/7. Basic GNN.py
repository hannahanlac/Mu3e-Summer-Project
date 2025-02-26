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
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def load_data(file_path):
    table = pq.read_table(file_path)  # Load Parquet file as an Arrow table
    print(table)
    awk_array = ak.from_arrow(table)  # Convert Arrow table to an Awkward Array
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
    # Prints the value associated with layer_array for hit indexed number 72 in frame 0 / batch 0

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
    print(f"Batch Number: {batch['frame_array'][0]}/{len(batches)}, Batch size: {len(batch['hit_ID'])}, Coords shape: {coords.shape}")

    # Ensure coords has shape (N, 3)
    if coords.shape[1] != 3:
        raise ValueError(f"Unexpected coordinate shape: {coords.shape}")

    G = nx.Graph()  # Initialize an empty graph

    # Add nodes
    for i, (x, y, z) in enumerate(coords):
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            print(f"Skipping node {i} due to NaN values")
            continue  # Skip nodes with NaN coordinates

        try:
            G.add_node(i, gx=x, gy=y, gz=z, hit_ID=batch['hit_ID'][i], tid=batch['tid_array'][i])
            # Node has features hit_ID and tid_array (need to ensure tid not accessible in training/testing)
        except KeyError as e:
            print(f"Missing key when adding node {i}: {e}")
            continue

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

for batch_graph in batch_graphs:
    print("Graph nodes with attributes:")
    for node, data in batch_graph.nodes(data=True):
        print(node, data)


print(f"Generated {len(batch_graphs)} graphs!")  
print("First batch graph details:", batch_graphs[0])  # Print first graph



'''def plot_graph3D(G):
    
    """Plots a 3D representation of the hit graph.

    Args:
        G: A NetworkX graph where nodes have 3D positions (gx, gy, gz)."""

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract node positions
    pos = {i: (G.nodes[i]['gx'], G.nodes[i]['gy'], G.nodes[i]['gz']) for i in G.nodes}

    # Plot nodes
    xs, ys, zs = zip(*pos.values())  # Unpack coordinates
    ax.scatter(xs, ys, zs, c='blue', marker='o', s=10, alpha=0.6, label="Hits")

    # Plot edges
    for edge in G.edges:
        x_vals = [pos[edge[0]][0], pos[edge[1]][0]]
        y_vals = [pos[edge[0]][1], pos[edge[1]][1]]
        z_vals = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x_vals, y_vals, z_vals, c='gray', alpha=0.4)  # Draw edge

    # Labels and styling
    ax.set_xlabel('gx')
    ax.set_ylabel('gy')
    ax.set_zlabel('gz')
    ax.set_title("Batch 0 3D Hit Graph Visualization")

    plt.legend()
    plt.show()

plot_graph3D(batch_graphs[0]) # Visualize the first batch's graph
'''

def extract_edge_features(G):
    """
    Extracts edge features directly from networkx graph.
    
    Args:
        G (networkx.Graph): The input graph.

    Returns:
        edge_features (np.ndarray): Feature matrix (E, 2C + C)
        edge_labels (np.ndarray): Ground truth labels (E, 1)
        edge_list (list): List of edges [(u, v), ...]
    """
    edge_features = []
    edge_labels = []
    edge_list = []

    for u, v in G.edges():
        """for node, attrs in G.nodes(data=True):
            print(node, attrs)"""  # Check what attributes are actually present

        if u not in G.nodes or v not in G.nodes:
            print(f"Warning: Edge ({u}, {v}) contains missing nodes")
            continue  # Skip missing nodes

        if 'gx' not in G.nodes[u] or 'gx' not in G.nodes[v]:
            print(f"Warning: Missing 'gx' attribute in nodes {u} or {v}")
            continue  # Skip edges where node attributes are missing



        # Get node features (assuming stored as attributes)
        source_feats = np.array([G.nodes[u]['gx'], G.nodes[u]['gy'], G.nodes[u]['gz']])
        target_feats = np.array([G.nodes[v]['gx'], G.nodes[v]['gy'], G.nodes[v]['gz']])
        
        # Compute feature difference
        feature_diff = source_feats - target_feats
        
        # Concatenate to form edge feature vector
        edge_feat = np.concatenate([source_feats, target_feats, feature_diff])  # (2C + C)

        # Check for missing 'tid' key before using it
        if 'tid' not in G.nodes[u] or 'tid' not in G.nodes[v]:
            print(f"Warning: Missing 'tid' attribute in nodes {u} or {v}")
            continue  # Skip this edge

        # Get ground truth labels (1 if same track, 0 otherwise)
        label = 1 if G.nodes[u]['tid'] == G.nodes[v]['tid'] else 0
        
        # Store results
        edge_features.append(edge_feat)
        edge_labels.append(label)
        edge_list.append((u, v))

    return np.array(edge_features), np.array(edge_labels).reshape(-1, 1), edge_list

def edge_classifier(edge_features, channels=(32, 64), with_bn=True, activation='relu', name='edge_classifier'):
    """
    MLP-based edge classifier for edge features.

    Args:
        edge_features: (E, 2C + C) Edge feature matrix
        channels: Tuple defining the MLP output sizes
        with_bn: Whether to use batch normalization
        activation: Activation function

    Returns:
        edge_logits: (E, 1) - Probability of being same track
    """
    inputs = keras.Input(shape=(edge_features.shape[1],))
    x = inputs
    
    for idx, channel in enumerate(channels):
        x = keras.layers.Dense(channel, activation=None, kernel_initializer='he_normal', name=f"{name}_dense{idx}")(x)

        if with_bn:
            x = keras.layers.BatchNormalization(name=f"{name}_bn{idx}")(x)
        
        x = keras.layers.Activation(activation, name=f"{name}_act{idx}")(x)

    # Final classification layer (probability of being the same track)
    edge_logits = keras.layers.Dense(1, activation='sigmoid', name=f"{name}_output")(x)

    return keras.Model(inputs, edge_logits)

# Extract edge features from one of our batch graphs
"""edge_features, edge_labels, edge_list = extract_edge_features(batch_graphs[0])

batch_edge_features = []
batch_edge_labels = []

for batch_graph in batch_graphs:
    edge_feats, edge_lbls, _ = extract_edge_features(batch_graph)
    batch_edge_features.append(edge_feats)
    batch_edge_labels.append(edge_lbls)

# Define model
model = edge_classifier(edge_features)

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(edge_features, edge_labels, epochs=10, batch_size=64)
# This is only working using the first batch, not all of them
"""

# Define model structure
edge_feats, _, _ = extract_edge_features(batch_graphs[0])  # Get first batch features
model = edge_classifier(np.zeros((1, edge_feats.shape[1])))  # Use the correct shape as dummy input to initialize

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model iteratively over batches
num_epochs = 10  # Define number of epochs

total_steps = num_epochs * len(batch_graphs)  # Total iterations across all epochs

with tqdm(total=total_steps, desc="Training Progress", unit="batch") as pbar:
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        for batch_graph in batch_graphs:
            # Extract edge features & labels for the batch
            edge_feats, edge_lbls, _ = extract_edge_features(batch_graph)

            # Convert to tensors
            edge_feats_tensor = tf.convert_to_tensor(edge_feats, dtype=tf.float32)
            edge_lbls_tensor = tf.convert_to_tensor(edge_lbls, dtype=tf.float32)

            # Train the model on this batch
            loss, acc = model.train_on_batch(edge_feats_tensor, edge_lbls_tensor)
            print(f"Batch loss: {loss:.4f}, Batch accuracy: {acc:.4f}")

            # Update the progress bar
            pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}", epoch=epoch+1)
            pbar.update(1)  # Move progress forward by one batch

print("Training complete! Saving model...")
model.save("trained_gnn_model")



#####################################
"""
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


################################################

"""def plot_graph2D(G):
    pos = {i: (G.nodes[i]['gx'], G.nodes[i]['gy']) for i in G.nodes}  # 2D projection
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_size=20, edge_color='gray', alpha=0.5)
    plt.xlabel('gx')
    plt.ylabel('gy')
    plt.title('Batch 0 Hit Graph')
    plt.show()

plot_graph2D(batch_graphs[0])"""
