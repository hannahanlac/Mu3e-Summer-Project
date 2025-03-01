import awkward as ak
import numpy as np
import tensorflow as tf
import pyarrow.parquet as pq
import networkx as nx
from scipy.spatial import cKDTree
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import pandas as pd


model_path = "trained_gnn_model"  # Path to saved model
model = tf.keras.models.load_model(model_path)

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

    print(batches[0][14]['layer_array']) 
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
    layers = ak.to_numpy(batch['layer_array'])
    hit_IDs = ak.to_numpy(batch['hit_ID'])

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
    for i, (x, y, z, layer, hit_ID) in enumerate(zip(gx, gy, gz, layers, hit_IDs)):
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            print(f"Skipping node {i} due to NaN values")
            continue  # Skip nodes with NaN coordinates

        try:
            G.add_node(i, gx=x, gy=y, gz=z, layer=layer, hit_ID=hit_ID)
            # Node has features hit_ID and tid_array (need to ensure tid not accessible in training/testing)
        except KeyError as e:
            print(f"Missing key when adding node {i}: {e}")
            continue

    # Find nearest neighbors for each hit
    tree = cKDTree(coords)  # KDTree for fast nearest neighbor search
    for i, coord in enumerate(coords):
        layer_i = layers[i]

        _, indices = tree.query(coord, k=k_neighbors + 1)  # +1 to exclude self

        for j in indices[1:]:  # Skip self (first index)
            if j >= len(layers):  # Ensure j is within valid bounds
                continue  

            layer_j = layers[j]  

            # Only allow connections to adjacent layers
            if abs(layer_j - layer_i) == 1:  # Ensure adjacency
                G.add_edge(i, j)  


    return G


file_path = 'ProcessedData/signal1_96_32652/test_data/test_data.parquet'
awk_data = load_data(file_path)
batches = batch_data(awk_data)
batch_graphs = [build_graph(batch) for batch in batches]

for batch_graph in batch_graphs:
    print("Graph nodes with attributes:")
    for node, data in batch_graph.nodes(data=True):
        print(node, data)


print(f"Generated {len(batch_graphs)} graphs!")  
print("First batch graph details:", batch_graphs[0])  # Print first graph

'''
def plot_graph3D(G):
    
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

plot_graph3D(batch_graphs[0])
'''
'''
def plot_graph2D(G):
    pos = {i: (G.nodes[i]['gx'].item(), G.nodes[i]['gy'].item()) for i in G.nodes}  # 2D projection
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_size=20, edge_color='gray', alpha=0.5)
    plt.xlabel('gx')
    plt.ylabel('gy')
    plt.title('Single Frame Hit Graph')
    plt.show()

plot_graph2D(batch_graphs[0])
plot_graph2D(batch_graphs[1])
plot_graph2D(batch_graphs[2])
plot_graph2D(batch_graphs[3])
plot_graph2D(batch_graphs[4])
plot_graph2D(batch_graphs[5])
'''









'''
def extract_edge_features(G):
    """Extracts edge features from graph without calculating ground truth labels liek we did in 7."""
    edge_features = []
    edge_list = []

    for u, v in G.edges():

        if u not in G.nodes or v not in G.nodes:
            print(f"Warning: Edge ({u}, {v}) contains missing nodes")
            continue  # Skip missing nodes

        if 'gx' not in G.nodes[u] or 'gx' not in G.nodes[v]:
            print(f"Warning: Missing 'gx' attribute in nodes {u} or {v}")
            continue  # Skip edges where node attributes are missing


        source_feats = np.array([G.nodes[u]['gx'], G.nodes[u]['gy'], G.nodes[u]['gz']])
        target_feats = np.array([G.nodes[v]['gx'], G.nodes[v]['gy'], G.nodes[v]['gz']])
        feature_diff = source_feats - target_feats

        edge_feat = np.concatenate([source_feats, target_feats, feature_diff])
        edge_features.append(edge_feat)
        edge_list.append((u, v))

    return np.array(edge_features), edge_list



# Store predictions in memory
batch_predictions = []

track_counter = 0  # Global counter for tracks

# Process batches and make predictions
with tqdm(total=len(batch_graphs), desc="Predicting Edges", unit="batch") as pbar:
    for batch_idx, batch in enumerate(batch_graphs):

        if len(batch_graph.edges) == 0:
            pbar.update(1)
            continue  

        edge_feats, edge_list = extract_edge_features(batch_graph)
        if len(edge_feats) == 0:
            pbar.update(1)
            continue  

        edge_feats_tensor = tf.convert_to_tensor(edge_feats, dtype=tf.float32)

        predictions = model.predict(edge_feats_tensor, verbose=0)

        # Store predictions in memory
        batch_predictions.append({
            "batch_idx": batch_idx,
            "edges": edge_list,
            "scores": predictions.flatten().tolist()
        })

        pbar.update(1)

print("Predictions stored in memory. Ready for post-processing!")

# Example: Access predictions for the first batch
print(f"Batch 0 Predictions: {batch_predictions[0]}")



def extract_tracks(batch_graphs, edge_scores):
    """
    Extracts 4-hit tracks from batch graphs while storing unmatched hits.

    Args:
        batch_graphs (list): List of NetworkX graphs representing batches.
        edge_scores (dict): Dictionary of edge scores { (node1, node2): score }.

    Returns:
        tuple: (list of 4-hit tracks, list of unmatched hits).
    """

    all_tracks = []  # Stores all valid 4-hit tracks
    unmatched_hits = []  # Stores hits that do not belong to any track

    for G in batch_graphs:  # Loop through each batch graph
        used_nodes = set()  # Track nodes that are already in a track
        node_layers = {}  # Dictionary to store nodes grouped by layer
        
        # Group nodes by their layer
        for node, data in G.nodes(data=True):
            layer = data['layer']
            if layer not in node_layers:
                node_layers[layer] = []
            node_layers[layer].append(node)

        # Sort layers to ensure correct sequence
        sorted_layers = sorted(node_layers.keys())  

        # Track-building process
        tracks = []
        for layer in sorted_layers[:-3]:  # Ensure at least 4 layers exist ahead
            for node in node_layers[layer]:
                if node in used_nodes:
                    continue  # Skip already used nodes

                track = [node]
                current_node = node
                valid_track = True

                for next_layer in sorted_layers[sorted_layers.index(layer) + 1:]:
                    candidates = [
                        neighbor for neighbor in G.neighbors(current_node)
                        if neighbor in node_layers[next_layer] and neighbor not in used_nodes
                    ]

                    if not candidates:
                        valid_track = False
                        break  # No valid hit found on the next layer

                    # Select the best connection based on edge scores
                    best_next_node = max(candidates, key=lambda n: edge_scores.get((current_node, n), 0))

                    track.append(best_next_node)
                    current_node = best_next_node

                    if len(track) == 4:
                        break  # Stop at 4-hit tracks

                if valid_track and len(track) == 4:
                    tracks.append(track)
                    used_nodes.update(track)  # Mark nodes as used

        # Collect unused hits that are not part of any track
        for layer in sorted_layers:
            for node in node_layers[layer]:
                if node not in used_nodes:
                    unmatched_hits.append(node)

        all_tracks.extend(tracks)

    return all_tracks, unmatched_hits

# Convert batch_predictions into a dictionary of edge scores
edge_scores_dict = { edge: score for edge, score in zip(batch_predictions['edges'], batch_predictions['scores']) }

# Extract tracks from predictions
tracks = extract_tracks(batch_graphs, batch_predictions)

def save_tracks_csv(tracks, output_path):
    """Saves extracted tracks to a CSV file for easy inspection."""
    track_data = []
    for batch in tracks:
        batch_idx = batch["batch_idx"]
        for track_id, track in enumerate(batch["tracks"]):
            for node in track:
                track_data.append((batch_idx, track_id, node))

    df = pd.DataFrame(track_data, columns=["batch_idx", "track_id", "node"])
    df.to_csv(output_path, index=False)


csv_output = "predicted_tracks.csv"

save_tracks_csv(tracks, csv_output)

print(f"Tracks saved to {csv_output}")
'''