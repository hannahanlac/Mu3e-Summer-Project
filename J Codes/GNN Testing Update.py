import awkward as ak
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from sklearn.utils import shuffle
import os
import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

model_path = "trained_gnn_model2"  # Path to saved model
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
    
    frame_values = ak.to_numpy(awk_array['frame_array'])  # Convert to NumPy
    print("Are frames sorted?", np.all(frame_values[:-1] <= frame_values[1:]))  # Check if sorted

    print(awk_array['layer_array'])

    return awk_array


file_path = 'ProcessedData/signal1_96_32652/test_data/test_data.parquet'
awk_data = load_data(file_path)


def batch_data(awk_array, frames_per_batch=1):
    unique_frames = np.unique(ak.to_numpy(awk_array['frame_array']))  # Get unique frame IDs
    batches = []

    for i in range(0, len(unique_frames), frames_per_batch):
        selected_frames = unique_frames[i:i + frames_per_batch]  # Select a chunk of frames
        mask = np.isin(ak.to_numpy(awk_array['frame_array']), selected_frames).flatten()  
        batch = awk_array[mask] 

        # Print batch summary
        print(f"Batch {len(batches) + 1}: {len(selected_frames)} frames, {len(batch['hit_ID'])} hits, {batch}")
        # Include the +1 because at time run this for i=0, len(batches) = 0 but we're currently building batch 1, therefore +1
        #print(f"Frames in batch: {selected_frames}")


        batches.append(batch)

    print(batches[0][13]['layer_array']) 
    # Prints the value associated with layer_array for hit indexed number 72 in frame 0 / batch 0

    return batches

batches = batch_data(awk_data)


def build_graph(batch, k_neighbours=5):
    """
    Constructs a graph for a batch based on spatial proximity (KNN).
    
    Args:
        batch: Awkward array containing hits for one batch.
        k_neighbours: Number of nearest neighbours to connect.

    Returns:
        G: A NetworkX graph representing the batch.
    """

    # Convert Awkward arrays to NumPy
    gx = ak.to_numpy(batch['gx'])
    gy = ak.to_numpy(batch['gy'])
    gz = ak.to_numpy(batch['gz'])
    layers = ak.to_numpy(batch['layer_array'])  # Convert layer_array to NumPy

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

    hitID_dict = {}

    # Add nodes
    for i, (x, y, z) in enumerate(coords):
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            print(f"Skipping node {i} due to NaN values")
            continue  # Skip nodes with NaN coordinates

        try:
            G.add_node(i, gx=x, gy=y, gz=z, 
            layer=batch['layer_array'][i], 
            station=batch['station_array'][i])
        
            # Store hit_ID separately for later evaluation (but not as part of the graph)
            hitID_dict[i] = batch['hit_ID'][i]  

        except KeyError as e:
            print(f"Missing key when adding node {i}: {e}")
            continue

####################


    # Dictionary grouping node indices by layer
    layer_indices = {layer: np.where(layers == layer)[0] for layer in np.unique(layers)}

    # KDTree for each layer for fast neighbour search
    kdtrees = {layer: cKDTree(coords[layer_indices[layer]]) for layer in layer_indices}

    # Define layer connection rules
    layer_connections = {
        1: [2],  # Layer 1 connects to Layer 2
        2: [1, 3],  # Layer 2 connects to Layer 1 and Layer 3
        3: [2, 4],  # Layer 3 connects to Layer 2 and Layer 4
        4: [3]  # Layer 4 connects to Layer 3
    }

    # Iterate through each layer and build edges with adjacent layers
    for layer, neighbours in layer_connections.items():
        if layer not in layer_indices:
            continue

        for neighbour_layer in neighbours:
            if neighbour_layer not in layer_indices:
                continue

            # Get the node indices for the current and adjacent layer
            layer_nodes = layer_indices[layer]
            neighbour_nodes = layer_indices[neighbour_layer]

            # Build KDTree for the neighbouring layer
            neighbour_tree = kdtrees[neighbour_layer]

            # Find k-nearest neighbours in the adjacent layer
            for i in layer_nodes:
                coord = coords[i]  # Get 3D position of the current node
                
                # Query k nearest neighbours in the adjacent layer
                k = min(k_neighbours, len(neighbour_nodes))  # Avoid querying more than available
                _, indices = neighbour_tree.query(coord, k=k)

                # Ensure indices is always an iterable (handle case where a single value is returned)
                if np.isscalar(indices):  
                    indices = np.array([indices]) # Convert single integer to array

                neighbour_indices = neighbour_nodes[indices]  # Map back to global node indices

                # Add edges
                for j in neighbour_indices:
                    G.add_edge(i, j)


    print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G, hitID_dict

batch_graphs = []
batch_hitID_dicts = []


for batch in batches:
    G, hitID_dict = build_graph(batch)  # Unpack both returned values
    batch_graphs.append(G)  # Store the graph
    batch_hitID_dicts.append(hitID_dict) # Store hitID_dict


# Print number of edges after graph construction
for i, G in enumerate(batch_graphs):
    print(f"Batch {i}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


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
        
        # Store results
        edge_features.append(edge_feat)
        edge_list.append((u, v))

    return np.array(edge_features), edge_list


#########


# Store predictions in memory
batch_predictions = []

track_counter = 0  # Global counter for tracks

# Process batches and make predictions
with tqdm(total=len(batch_graphs), desc="Predicting Edges", unit="batch") as pbar:
    for batch_idx, batch in enumerate(batch_graphs):

        if len(batch_graphs[batch_idx].edges) == 0:
            pbar.update(1)
            continue  

        edge_feats, edge_list = extract_edge_features(batch_graphs[batch_idx])
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
print(f"Batch 1 Predictions: {batch_predictions[1]}")
print(f"Batch 2 Predictions: {batch_predictions[2]}")