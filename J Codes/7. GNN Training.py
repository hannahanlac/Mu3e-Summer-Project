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
    print("First few traj_pt values", ak.to_numpy(awk_array['traj_pt'])[:10])
    
    frame_values = ak.to_numpy(awk_array['frame_array'])  # Convert to NumPy
    print("Are frames sorted?", np.all(frame_values[:-1] <= frame_values[1:]))  # Check if sorted

    print(awk_array['layer_array'])

    return awk_array


file_path = '/users/gy22186/mu3e/GNNtrain_data/train_data.parquet'
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

    print(batches[0][72]['layer_array']) 
    # Prints the value associated with layer_array for hit indexed number 72 in frame 0 / batch 0

    return batches

batches = batch_data(awk_data)

selected_hit = batches[0][13]
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
    tid_dict = {} # Create separate dictionary to store so can't access in training (i.e., not node feature)
    traj_p_dict = {}
    traj_pt_dict = {}
    traj_lambda_dict = {}
    traj_phi_dict = {}

    # Add nodes
    for i, (x, y, z) in enumerate(coords):
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            print(f"Skipping node {i} due to NaN values")
            continue  # Skip nodes with NaN coordinates

        try:
            G.add_node(i, gx=x, gy=y, gz=z, 
            layer=batch['layer_array'][i], 
            station=batch['station_array'][i],
            ladder=batch['ladder_array'][i],
            chip=batch['chip_array'][i])
        
            # Store tid separately for later evaluation (but NOT as part of the graph)
            tid_dict[i] = batch['tid_array'][i]
            hitID_dict[i] = batch['hit_ID'][i]
            traj_p_dict[i] = batch['traj_p'][i]
            traj_pt_dict[i] = batch['traj_pt'][i]
            traj_lambda_dict[i] = batch['traj_lambda'][i]
            traj_phi_dict[i] = batch['traj_phi'][i]

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

    return G, hitID_dict, tid_dict, traj_p_dict, traj_pt_dict, traj_lambda_dict, traj_phi_dict

batch_graphs = []
batch_hitID_dicts = []
batch_tid_dicts = []
batch_p_dicts = []
batch_pt_dicts = []
batch_lambda_dicts = []
batch_phi_dicts = []


for batch in batches:
    G, hitID_dict, tid_dict, traj_p_dict, traj_pt_dict, traj_lambda_dict, traj_phi_dict = build_graph(batch)  # Unpack both returned values
    batch_graphs.append(G)  # Store the graph
    batch_hitID_dicts.append(hitID_dict) # Store hitID_dict
    batch_tid_dicts.append(tid_dict)  # Store tid_dict
    batch_p_dicts.append(traj_p_dict)
    batch_pt_dicts.append(traj_pt_dict)
    batch_lambda_dicts.append(traj_lambda_dict)
    batch_phi_dicts.append(traj_phi_dict)


# Print number of edges after graph construction
for i, G in enumerate(batch_graphs):
    print(f"Batch {i}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")




def plot_graph2D(G):
    pos = {i: (G.nodes[i]['gx'], G.nodes[i]['gy']) for i in G.nodes}  # 2D projection
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_size=20, edge_color='gray', alpha=0.5)
    plt.xlabel('gx')
    plt.ylabel('gy')
    plt.title('Frame 0 Hit Graph')
    plt.show()

for i in range(min(3, len(batch_graphs))):
    plot_graph2D(batch_graphs[i])

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

for i in range(min(3, len(batch_graphs))):
    plot_graph3D(batch_graphs[i])

'''
def extract_edge_features(G, tid_dict):
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

        if u not in tid_dict or v not in tid_dict:
            print(f"Warning: Missing 'tid' in tid_dict for nodes {u} or {v}")
            continue  # Skip this edge     

        # Get ground truth labels (1 if same track, 0 otherwise)
        label = 1 if tid_dict[u] == tid_dict[v] else 0
        #print(f"Edge ({u}, {v}): tid_u = {tid_dict[u]}, tid_v = {tid_dict[v]}")
        #print(label)
        
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

# Define model structure
edge_feats, _, _ = extract_edge_features(batch_graphs[0], batch_tid_dicts[0])  # Get first batch features
model = edge_classifier(np.zeros((1, edge_feats.shape[1])))  # Use the correct shape as dummy input to initialize

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model iteratively over batches
num_epochs = 10  # Define number of epochs

total_steps = num_epochs * len(batch_graphs)  # Total iterations across all epochs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

with tqdm(total=total_steps, desc="Training Progress", unit="batch") as pbar:
    for epoch in range(num_epochs):
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        epoch_losses = []  # Store batch losses for averaging
        epoch_accuracies = []

        for batch_graph, tid_dict in zip(batch_graphs, batch_tid_dicts):
            # Extract edge features & labels for the batch
            edge_feats, edge_lbls, _ = extract_edge_features(batch_graph, tid_dict)

            # Skip empty batches
            if edge_feats.shape[0] == 0 or edge_lbls.shape[0] == 0:
                tqdm.write(f"Skipping empty batch at epoch {epoch+1}")
                continue

            # Convert to tensors
            edge_feats_tensor = tf.convert_to_tensor(edge_feats, dtype=tf.float32)
            edge_lbls_tensor = tf.convert_to_tensor(edge_lbls, dtype=tf.float32)

            # Train the model on this batch
            loss, acc = model.train_on_batch(edge_feats_tensor, edge_lbls_tensor)
            #tqdm.write(f"Batch loss: {loss:.4f}, Batch accuracy: {acc:.4f}")

            epoch_losses.append(loss)
            epoch_accuracies.append(acc)

            # Update the progress bar
            pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}", epoch=epoch+1)
            pbar.update(1)  # Move progress forward by one batch

        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accuracies)
        tqdm.write(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")


print("Training complete! Saving model...")
model.save("trained_gnn_model3")
'''