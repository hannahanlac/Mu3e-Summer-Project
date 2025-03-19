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
import logging
import time

def load_data(file_path):
    table = pq.read_table(file_path)  # Load Parquet file as an Arrow table
    print(table)
    awk_array = ak.from_arrow(table)  # Convert Arrow table to an Awkward Array
    print(awk_array)

    # Convert Awkward arrays to flat Numpy arrays
    flat_dict = {key: ak.to_numpy(awk_array[key]).flatten() for key in awk_array.fields}

    # Convert to a new Awkward array (all values now 1D)
    awk_array = ak.Array(flat_dict)

    # Debugging: Check the first few rows
    print(f"Loaded dataset: {len(awk_array['hit_ID'])} hits")

    print("First few frame values:", ak.to_numpy(awk_array['frame_array'])[:10])
    print("Frames from index 5000-5020:", ak.to_numpy(awk_array['frame_array'])[5000:5020])

    print("First few hit IDs:", ak.to_numpy(awk_array['hit_ID'])[:10])
    print("First few traj_pt values", ak.to_numpy(awk_array['traj_pt'])[:10])
    
    frame_values = ak.to_numpy(awk_array['frame_array'])  # Convert to NumPy
    print("Are frames sorted?", np.all(frame_values[:-1] <= frame_values[1:]))  # Check if sorted

    print(awk_array['layer_array'])

    return awk_array


file_path = 'ProcessedData/signal1_95-99_32652/train_data/train_data.parquet'
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

        batches.append(batch)

    print(batches[0][5]['layer_array']) 
    # Prints the value associated with layer_array for hit indexed number 72 in frame 0 / batch 0

    return batches

batches = batch_data(awk_data)

selected_hit = batches[0][7]
print(f"Hit 13 in Batch 0:")
for key in selected_hit.fields:  # Iterate over all feature names
    print(f"{key}: {selected_hit[key]}")


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
    stations = ak.to_numpy(batch['station_array'])
    ladders = ak.to_numpy(batch['ladder_array'])
    chips = ak.to_numpy(batch['chip_array'])
    hit_IDs = ak.to_numpy(batch['hit_ID'])
    tids = ak.to_numpy(batch['tid_array'])
    traj_p = ak.to_numpy(batch['traj_p'])
    traj_pt = ak.to_numpy(batch['traj_pt'])
    traj_lambda = ak.to_numpy(batch['traj_lambda'])
    traj_phi = ak.to_numpy(batch['traj_phi'])

    # Ensure no NaN values
    valid_mask = ~(np.isnan(gx) | np.isnan(gy) | np.isnan(gz))
    
    gx, gy, gz = gx[valid_mask], gy[valid_mask], gz[valid_mask]
    layers, stations, ladders, chips = layers[valid_mask], stations[valid_mask], ladders[valid_mask], chips[valid_mask]
    hit_IDs, tids, traj_p, traj_pt, traj_lambda, traj_phi = hit_IDs[valid_mask], tids[valid_mask], traj_p[valid_mask], traj_pt[valid_mask], traj_lambda[valid_mask], traj_phi[valid_mask]

    # Stack into coordinate array
    coords = np.column_stack((gx, gy, gz))  # Alternative to vstack.T, shape (N, 3)

    # Debugging: Print the shape of the coordinates
    print(f"Batch Number: {batch['frame_array'][0]}/{len(batches)}, Batch size: {len(batch['hit_ID'])}, Coords shape: {coords.shape}")

    # Ensure coords has shape (N, 3)
    if coords.shape[1] != 3:
        raise ValueError(f"Unexpected coordinate shape: {coords.shape}")

    G = nx.Graph()  # Initialize an empty graph

    node_truth_info = {
        i: {
            "hit_ID": hit_IDs[i],
            "tid": tids[i],
            "traj_p": traj_p[i],
            "traj_pt": traj_pt[i],
            "traj_lambda": traj_lambda[i],
            "traj_phi": traj_phi[i]
        }
        for i in range(len(coords))
    }

    # Add nodes
    for i, (x, y, z, layer, station, ladder, chip) in enumerate(zip(gx, gy, gz, layers, stations, ladders, chips)):
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            print(f"Skipping node {i} due to NaN values")
            continue  # Skip nodes with NaN coordinates

        try:
            G.add_node(i, gx=x, gy=y, gz=z, layer=layer, station=station, ladder=ladder, chip=chip)
        except KeyError as e:
            print(f"Missing key when adding node {i}: {e}")
            continue

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

        layer_nodes = layer_indices[layer] # Get node indices for current layer

        for neighbour_layer in neighbours:
            if neighbour_layer not in layer_indices:
                continue

            # Get node indices for adjacent layer
            neighbour_nodes = layer_indices[neighbour_layer]

            # Build KDTree for the neighbouring layer
            neighbour_tree = kdtrees[neighbour_layer]

            # Find k-nearest neighbours in the adjacent layer
            for i in layer_nodes:
                coord = coords[i]  # Get 3D position of the current node
                
                distances, indices = neighbour_tree.query(coord, k=min(k_neighbours, len(neighbour_nodes)))

                indices = np.atleast_1d(indices)  # Ensure indices is always iterable

                # Vectorized mapping of indices
                for idx in indices:
                    G.add_edge(i, neighbour_nodes[idx])

    print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G, node_truth_info


def extract_edge_features(G, node_truth_info):
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
        if u not in G.nodes or v not in G.nodes:
            print(f"Warning: Edge ({u}, {v}) contains missing nodes")
            continue  # Skip missing nodes

        required_attrs = ['gx', 'gy', 'gz', 'layer', 'station', 'ladder', 'chip']
        if any(attr not in G.nodes[u] or attr not in G.nodes[v] for attr in required_attrs):
            print(f"Warning: Missing attributes in nodes {u} or {v}")
            continue

        # Get node features (assuming stored as attributes)
        source_feats = np.array([G.nodes[u]['gx'], G.nodes[u]['gy'], G.nodes[u]['gz']])
        target_feats = np.array([G.nodes[v]['gx'], G.nodes[v]['gy'], G.nodes[v]['gz']])
        
        # Compute feature difference
        feature_diff = source_feats - target_feats

        categorical_feats = np.array([
            G.nodes[u]['layer'], G.nodes[u]['station'], G.nodes[u]['ladder'], G.nodes[u]['chip'],
            G.nodes[v]['layer'], G.nodes[v]['station'], G.nodes[v]['ladder'], G.nodes[v]['chip']
        ])
        
        # Concatenate to form edge feature vector
        edge_feat = np.concatenate([source_feats, target_feats, feature_diff, categorical_feats])  # (2C + C)

        if u not in node_truth_info or v not in node_truth_info:
            print(f"Warning: Missing truth info for nodes {u} or {v}")
            continue   

        # Get ground truth labels (1 if same track, 0 otherwise)
        label = 1 if node_truth_info[u]['tid'] == node_truth_info[v]['tid'] else 0
        
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


# Determine the number of epochs based on the old training time
# If 10 epochs for the old file took 30 minutes, and the new file is 5 times larger:
old_training_time_per_epoch = 30 / 10  # minutes per epoch
new_training_time_per_epoch = old_training_time_per_epoch * 5  # estimated

# Let's assume you want to train for a maximum of 8 hours (480 minutes)
max_training_time = 450  # in minutes
num_epochs = int(max_training_time / new_training_time_per_epoch)

print(f"Estimated number of epochs: {num_epochs}")

# Define model structure
edge_feats, _, _ = extract_edge_features(build_graph(batches[0])[0], build_graph(batches[0])[1])  # Get first batch features
model = edge_classifier(np.zeros((1, edge_feats.shape[1])))  # Use the correct shape as dummy input to initialize

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add EarlyStopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='loss',  # Monitor the loss
    patience=5,      # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the best weights after stopping
)

# Train model iteratively over batches
total_steps = num_epochs * len(batches)  # Total iterations across all epochs
best_loss = np.inf
patience_counter = 0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

with tqdm(total=total_steps, desc="Training Progress", unit="batch") as pbar:
    for epoch in range(num_epochs):
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        epoch_losses = []  # Store batch losses for averaging
        epoch_accuracies = []

        for batch in batches:
            # Build graph dynamically for each batch
            G, node_truth_info = build_graph(batch)

            # Extract edge features & labels
            edge_feats, edge_lbls, _ = extract_edge_features(G, node_truth_info)

            # Skip empty batches
            if edge_feats.shape[0] == 0 or edge_lbls.shape[0] == 0:
                tqdm.write(f"Skipping empty batch at epoch {epoch+1}")
                continue

            # Convert to tensors
            edge_feats_tensor = tf.convert_to_tensor(edge_feats, dtype=tf.float32)
            edge_lbls_tensor = tf.convert_to_tensor(edge_lbls, dtype=tf.float32)

            # Train the model on this batch
            loss, acc = model.train_on_batch(edge_feats_tensor, edge_lbls_tensor)

            epoch_losses.append(loss)
            epoch_accuracies.append(acc)

            # Update progress bar
            pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}", epoch=epoch+1)
            pbar.update(1)

        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accuracies)
        tqdm.write(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")

        # Check for early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        if patience_counter >= early_stopping.patience:
            tqdm.write("Early stopping triggered. Restoring best model weights and stopping training.")
            model.set_weights(early_stopping.best_weights)
            break

print("Training complete! Saving model...")
model.save("trained_gnn_model6")

