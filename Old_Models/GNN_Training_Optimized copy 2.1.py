import awkward as ak
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import pyarrow.parquet as pq
import networkx as nx
from scipy.spatial import cKDTree
from tqdm import tqdm
import logging

def load_data(file_path):
    table = pq.read_table(file_path)
    awk_array = ak.from_arrow(table)

    # Flatten nested arrays to 1D
    flat_dict = {key: ak.to_numpy(awk_array[key]).flatten() for key in awk_array.fields}
    awk_array = ak.Array(flat_dict)

    print(f"Loaded dataset: {len(awk_array['hit_ID'])} hits")
    return awk_array

def build_graph(frame_hits, k_neighbours=5):
    # Convert fields to numpy arrays
    gx = ak.to_numpy(frame_hits['gx'])
    gy = ak.to_numpy(frame_hits['gy'])
    gz = ak.to_numpy(frame_hits['gz'])
    layers = ak.to_numpy(frame_hits['layer_array'])
    stations = ak.to_numpy(frame_hits['station_array'])
    ladders = ak.to_numpy(frame_hits['ladder_array'])
    chips = ak.to_numpy(frame_hits['chip_array'])
    hit_IDs = ak.to_numpy(frame_hits['hit_ID'])
    tids = ak.to_numpy(frame_hits['tid_array'])
    traj_p = ak.to_numpy(frame_hits['traj_p'])
    traj_pt = ak.to_numpy(frame_hits['traj_pt'])
    traj_lambda = ak.to_numpy(frame_hits['traj_lambda'])
    traj_phi = ak.to_numpy(frame_hits['traj_phi'])

    # Filter out NaNs
    valid_mask = ~(np.isnan(gx) | np.isnan(gy) | np.isnan(gz))
    gx, gy, gz = gx[valid_mask], gy[valid_mask], gz[valid_mask]
    layers, stations, ladders, chips = layers[valid_mask], stations[valid_mask], ladders[valid_mask], chips[valid_mask]
    hit_IDs, tids = hit_IDs[valid_mask], tids[valid_mask]
    traj_p, traj_pt, traj_lambda, traj_phi = traj_p[valid_mask], traj_pt[valid_mask], traj_lambda[valid_mask], traj_phi[valid_mask]

    coords = np.column_stack((gx, gy, gz))
    if coords.shape[0] == 0:
        return None, None  # Skip empty frame

    G = nx.Graph()

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
        G.add_node(i, gx=x, gy=y, gz=z, layer=layer, station=station, ladder=ladder, chip=chip)

    # Group node indices by layer
    layer_indices = {layer: np.where(layers == layer)[0] for layer in np.unique(layers)}

    kdtrees = {layer: cKDTree(coords[layer_indices[layer]]) for layer in layer_indices}

    layer_connections = {
        1: [2],
        2: [1, 3],
        3: [2, 4],
        4: [3]
    }

    for layer, neighbours in layer_connections.items():
        if layer not in layer_indices:
            continue

        layer_nodes = layer_indices[layer]

        for neighbour_layer in neighbours:
            if neighbour_layer not in layer_indices:
                continue

            neighbour_nodes = layer_indices[neighbour_layer]
            neighbour_tree = kdtrees[neighbour_layer]  

            for i in layer_nodes:
                if i >= len(coords):
                    continue  # Defensive check

                # Query nearest neighbors
                distances, indices = neighbour_tree.query(coords[i], k=min(k_neighbours, len(neighbour_nodes)))
                if np.isscalar(indices):
                    indices = [indices]

                for idx in indices:
                    j = neighbour_nodes[idx]
                    G.add_edge(i, j)

    return G, node_truth_info

def extract_edge_features(G, node_truth_info):
    edge_features = []
    edge_labels = []
    edge_list = []

    for u, v in G.edges():
        source_feats = np.array([G.nodes[u]['gx'], G.nodes[u]['gy'], G.nodes[u]['gz']])
        target_feats = np.array([G.nodes[v]['gx'], G.nodes[v]['gy'], G.nodes[v]['gz']])
        feature_diff = source_feats - target_feats

        categorical_feats = np.array([
            G.nodes[u]['layer'], G.nodes[u]['station'], G.nodes[u]['ladder'], G.nodes[u]['chip'],
            G.nodes[v]['layer'], G.nodes[v]['station'], G.nodes[v]['ladder'], G.nodes[v]['chip']
        ])

        edge_feat = np.concatenate([source_feats, target_feats, feature_diff, categorical_feats])

        # Label 1 if same track, else 0
        label = 1 if node_truth_info[u]['tid'] == node_truth_info[v]['tid'] else 0

        edge_features.append(edge_feat)
        edge_labels.append(label)
        edge_list.append((u, v))

    return np.array(edge_features), np.array(edge_labels).reshape(-1, 1), edge_list

def edge_classifier(input_shape, channels=(32, 64), with_bn=True, activation='relu', name='edge_classifier'):
    inputs = keras.Input(shape=(input_shape,))
    x = inputs
    for idx, channel in enumerate(channels):
        x = keras.layers.Dense(channel, activation=None, kernel_initializer='he_normal', name=f"{name}_dense{idx}")(x)
        if with_bn:
            x = keras.layers.BatchNormalization(name=f"{name}_bn{idx}")(x)
        x = keras.layers.Activation(activation, name=f"{name}_act{idx}")(x)
    edge_logits = keras.layers.Dense(1, activation='sigmoid', name=f"{name}_output")(x)
    return keras.Model(inputs, edge_logits)

# Load data
file_path = '/users/gy22186/mu3e/five_signal_files/train_data.parquet'
awk_data = load_data(file_path)

# Get all unique frames
all_frames = np.unique(ak.to_numpy(awk_data['frame_array']))

# Build one graph per frame
frame_graphs = []
frame_truth_info = []

print("Building graphs for each frame...")
for frame_id in tqdm(all_frames):
    frame_mask = (ak.to_numpy(awk_data['frame_array']) == frame_id)
    frame_hits = awk_data[frame_mask]
    if len(frame_hits['hit_ID']) == 0:
        continue
    G, truth_info = build_graph(frame_hits)
    if G is not None:
        frame_graphs.append(G)
        frame_truth_info.append(truth_info)

print(f"Built {len(frame_graphs)} graphs (one per frame).")

# Batch graphs (e.g. 8 graphs per batch)
batch_size = 5000
graph_batches = [frame_graphs[i:i+batch_size] for i in range(0, len(frame_graphs), batch_size)]
truth_batches = [frame_truth_info[i:i+batch_size] for i in range(0, len(frame_truth_info), batch_size)]

# Initialize model with dummy input to get input shape
dummy_feat = np.zeros((1, 17))  # 14 is the length of edge feature vector (3+3+3+8)
model = edge_classifier(dummy_feat.shape[1])
# Set a custom learning rate
learning_rate = 0.4  # Adjust this value as needed
optimizer = Adam(learning_rate=learning_rate)
# Compile the model with the custom optimizer
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training parameters
num_epochs = 20
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

total_steps = num_epochs * len(graph_batches)

with tqdm(total=total_steps, desc="Training Progress", unit="batch") as pbar:
    for epoch in range(num_epochs):
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
        epoch_losses = []
        epoch_accuracies = []

        for batch_graphs, batch_truths in zip(graph_batches, truth_batches):
            batch_edge_feats = []
            batch_edge_labels = []

            # Collect edges from all graphs in the batch
            for G, truth_info in zip(batch_graphs, batch_truths):
                edge_feats, edge_lbls, _ = extract_edge_features(G, truth_info)
                if edge_feats.shape[0] == 0:
                    continue
                batch_edge_feats.append(edge_feats)
                batch_edge_labels.append(edge_lbls)

            if not batch_edge_feats:
                tqdm.write(f"Skipping empty batch at epoch {epoch+1}")
                continue

            # Concatenate edge features and labels from all graphs
            batch_edge_feats = np.vstack(batch_edge_feats)
            batch_edge_labels = np.vstack(batch_edge_labels)

            edge_feats_tensor = tf.convert_to_tensor(batch_edge_feats, dtype=tf.float32)
            edge_lbls_tensor = tf.convert_to_tensor(batch_edge_labels, dtype=tf.float32)

            loss, acc = model.train_on_batch(edge_feats_tensor, edge_lbls_tensor)
            epoch_losses.append(loss)
            epoch_accuracies.append(acc)

            pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}", epoch=epoch+1)
            pbar.update(1)

        avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        avg_acc = np.mean(epoch_accuracies) if epoch_accuracies else float('nan')
        tqdm.write(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")

print("Training complete! Saving model...")
model.save("trained_gnn_model_five_signals.keras")
