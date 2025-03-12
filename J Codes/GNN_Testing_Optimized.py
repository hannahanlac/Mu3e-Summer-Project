import awkward as ak
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from sklearn.utils import shuffle
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import logging

model_path = "trained_gnn_model4"  # Path to saved model
model = tf.keras.models.load_model(model_path)


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


file_path = 'ProcessedData/signal1_96_32652/train_data/train_data.parquet'
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
    #print(f"Batch Number: {batch['frame_array'][0]}/{len(batches)}, Batch size: {len(batch['hit_ID'])}, Coords shape: {coords.shape}")

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
                
            distances, indices = neighbour_tree.query(coords[layer_nodes], k=min(k_neighbours, len(neighbour_nodes)))

            if k_neighbours == 1:
                indices = np.atleast_2d(indices).T

            # Vectorized mapping of indices
            for i, idx_list in zip(layer_nodes, indices):
                idx_list = np.atleast_1d(idx_list)
                for j in neighbour_nodes[idx_list]:
                    G.add_edge(i, j)


    print(f"Graph Batch Number: {batch['frame_array'][0]}/{len(batches)-1}, {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G, node_truth_info

batch_graphs = []
batch_truth_info = []


for batch in batches:
    G, truth_info = build_graph(batch)  # Unpack both returned values
    batch_graphs.append(G)  # Store the graph
    batch_truth_info.append(truth_info)

print(batch_truth_info[0][5])  # Truth data for node 5 in batch 0


# Print number of edges after graph construction
"""for i, G in enumerate(batch_graphs):
    print(f"Batch {i}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")"""


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
        """for node, attrs in G.nodes(data=True):
            print(node, attrs)"""  # Check what attributes are actually present

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


#########

# Store predictions in memory
batch_predictions = []

track_counter = 0  # Global counter for tracks

precomputed_edges = []
precomputed_edge_lists = []

with tqdm(total=len(batch_graphs), desc="Extracting Edge Features", unit="batch") as pbar:
    for batch_idx, batch in enumerate(batch_graphs):

        edge_feats, _, edge_list = extract_edge_features(batch, batch_truth_info[batch_idx])

        precomputed_edges.append(edge_feats)
        precomputed_edge_lists.append(edge_list)

        pbar.update(1)

batch_edges_np = [np.array(edges, dtype=np.float32) for edges in precomputed_edges]

# Process batches and make predictions
with tqdm(total=len(batch_graphs), desc="Predicting Edges", unit="batch") as pbar:

        # Iterate over batches
    for batch_idx, edge_feats in enumerate(batch_edges_np):

        # Ensure predictions are stored even if empty
        if edge_feats.shape[0] == 0:
            batch_predictions.append({
                "batch_idx": batch_idx,
                "edges": [],
                "scores": []
            })
            
        else:
            predictions = model.predict_on_batch(edge_feats)

            batch_predictions.append({
                "batch_idx": batch_idx,
                "edges": precomputed_edge_lists[batch_idx],
                "scores": predictions.flatten().tolist()
            })

        pbar.update(1)

print("Predictions stored in memory. Ready for post-processing!")

# Example: Access predictions for the first batch
print(f"Batch 0 Predictions: {batch_predictions[0]}")
print(f"Batch 1 Predictions: {batch_predictions[1]}")
print(f"Batch 2 Predictions: {batch_predictions[2]}")



def construct_tracks(batch_graphs, batch_predictions, batch_truth_info, min_track_size=4):
    """
    Constructs tracks from batch predictions, ensuring each track has one hit per layer.
    
    Args:
        batch_graphs (list): List of NetworkX graphs for each batch.
        batch_predictions (list): List of dictionaries containing edge scores.
        batch_hitID_dicts (list): List of hit_ID dictionaries per batch.
        min_track_size (int): Minimum number of hits in a valid track.

    Returns:
        batch_tracks (list): List of tracks for each batch.
    """
    batch_tracks = []

    for batch_idx, pred in enumerate(batch_predictions):
        edges, scores = pred["edges"], pred["scores"]
        G = batch_graphs[batch_idx]  # Get the corresponding graph 
        truth_info = batch_truth_info[batch_idx]

        tracks = []  # List to store tracks

        # Sort edges by descending score
        scored_edges = sorted(zip(edges, scores), key=lambda x: x[1], reverse=True)

        for (u, v), score in scored_edges:
            if score < 0.9:  # Threshold for valid edges (tune as needed)
                continue

            # DEBUG: Check if nodes exist in G before accessing them
            if u not in G.nodes or v not in G.nodes:
                print(f"Warning: Node {u} or {v} not found in batch {batch_idx}. Skipping this edge.")
                continue  # Skip invalid edges

            # Get layers of the two nodes
            layer_u = G.nodes[u]["layer"]
            layer_v = G.nodes[v]["layer"]

            hitID_u = truth_info[u]["hit_ID"]
            hitID_v = truth_info[v]["hit_ID"]
            tid_u = truth_info[u]["tid"]
            tid_v = truth_info[v]["tid"]
            traj_p_u = truth_info[u]["traj_p"]
            traj_p_v = truth_info[v]["traj_p"]
            traj_pt_u = truth_info[u]["traj_pt"]
            traj_pt_v = truth_info[v]["traj_pt"]
            traj_lambda_u = truth_info[u]["traj_lambda"]
            traj_lambda_v = truth_info[v]["traj_lambda"]
            traj_phi_u = truth_info[u]["traj_phi"]
            traj_phi_v = truth_info[v]["traj_phi"]

            # Find an existing track that can include this edge
            added = False
            for track in tracks:
                track_layers = {G.nodes[n]["layer"] for n in track["nodes"]}
                if layer_u not in track_layers or layer_v not in track_layers:
                    track["nodes"].append(u)
                    track["nodes"].append(v)
                    track["hit_IDs"].append(hitID_u)
                    track["hit_IDs"].append(hitID_v)
                    track["tids"].extend([int(tid_u), int(tid_v)])
                    track["traj_ps"].extend([int(traj_p_u), int(traj_p_v)])
                    track["traj_pts"].extend([int(traj_pt_u), int(traj_pt_v)])
                    track["traj_lambdas"].extend([int(traj_lambda_u), int(traj_lambda_v)])
                    track["traj_phis"].extend([int(traj_phi_u), int(traj_phi_v)])
                    added = True
                    break

            # If no existing track can accommodate this edge, create a new track
            if not added:
                tracks.append({
                    "nodes": [u, v], 
                    "hit_IDs": [hitID_u, hitID_v], 
                    "tids": [tid_u, tid_v],
                    "traj_ps": [traj_p_u, traj_p_v],
                    "traj_pts": [traj_pt_u, traj_pt_v],
                    "traj_lambdas": [traj_lambda_u, traj_lambda_v],
                    "traj_phis": [traj_phi_u, traj_phi_v]
                    })

        # Filter tracks to ensure they have at least one hit per layer and min size
        valid_tracks = [track for track in tracks if len(track["nodes"]) >= min_track_size]

        batch_tracks.append(valid_tracks)

        print(f"Batch {batch_idx}: {len(valid_tracks)} valid tracks found")

    return batch_tracks


# Run the function to extract valid tracks
batch_tracks = construct_tracks(batch_graphs, batch_predictions, batch_truth_info)



def save_tracks(batch_tracks, output_dir="output"):
    """
    Saves track data to CSV and Parquet format.

    Args:
        batch_tracks (list): List of tracks for each batch.
        output_dir (str): Directory to save output files.
    """

    all_tracks = []

    for batch_idx, tracks in enumerate(batch_tracks):
        truth_info = batch_truth_info[batch_idx]

        for track_idx, track in enumerate(tracks):
            track_tids = track["tids"]

            correct_track = 1 if all(tid == track_tids[0] for tid in track_tids) else 0

            all_tracks.append({
                "batch": batch_idx,
                "track_id": track_idx,
                "hit_IDs": track["hit_IDs"],
                "tids": track["tids"],
                "traj_ps": track["traj_ps"],
                "traj_pts": track["traj_pts"],
                "traj_lambdas": track["traj_lambdas"],
                "traj_phis": track["traj_phis"],
                "num_hits": len(track["hit_IDs"]),
                "correct_track": correct_track,
            })

    # Convert to DataFrame
    df = pd.DataFrame(all_tracks)

    # Save to CSV
    csv_path = "ProcessedData/signal1_96_32652/reconstrcted_tracks/predicted_tracks3.csv"
    df.to_csv(csv_path, index=False)

    # Save to Parquet
    parquet_path = "ProcessedData/signal1_96_32652/reconstrcted_tracks/predicted_tracks3.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)

    print(f"Tracks saved to {csv_path} and {parquet_path}")


# Save the tracks
save_tracks(batch_tracks)
