import awkward as ak
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
import os
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from tqdm import tqdm
import logging

import cudf
import cugraph

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow is using:", tf.config.list_physical_devices('GPU'))

def load_data(file_path):
    table = pq.read_table(file_path)
    awk_array = ak.from_arrow(table)
    flat_dict = {key: ak.to_numpy(awk_array[key]).flatten() for key in awk_array.fields}
    awk_array = ak.Array(flat_dict)
    print(f"Loaded dataset: {len(awk_array['hit_ID'])} hits")
    return awk_array

def batch_data_generator(awk_array, frames_per_batch=1):
    frame_array_np = ak.to_numpy(awk_array['frame_array'])
    unique_frames = np.unique(frame_array_np)
    for i in range(0, len(unique_frames), frames_per_batch):
        selected_frames = unique_frames[i:i + frames_per_batch]
        mask = np.isin(frame_array_np, selected_frames).flatten()
        batch = awk_array[mask]
        yield batch

def build_graph_cugraph_gpu(batch, k_neighbours=5):
    # Convert batch to cuDF DataFrame
    df = cudf.DataFrame({
        'gx': batch['gx'],
        'gy': batch['gy'],
        'gz': batch['gz'],
        'layer': batch['layer_array'],
        'station': batch['station_array'],
        'ladder': batch['ladder_array'],
        'chip': batch['chip_array'],
        'hit_ID': batch['hit_ID'],
        'tid': batch['tid_array'],
        'traj_p': batch['traj_p'],
        'traj_pt': batch['traj_pt'],
        'traj_lambda': batch['traj_lambda'],
        'traj_phi': batch['traj_phi']
    })

    # Drop NaNs
    df = df.dropna()

    # Assign node IDs (needed for edges)
    df = df.reset_index(drop=True)
    df['node_id'] = cp.arange(len(df))

    # Get unique layers
    unique_layers = df['layer'].unique().to_pandas().to_list()
    layer_connections = {1: [2], 2: [1, 3], 3: [2, 4], 4: [3]}

    src_nodes = []
    dst_nodes = []

    for layer in unique_layers:
        if layer not in layer_connections:
            continue
        df_layer = df[df['layer'] == layer]
        coords_layer = cp.stack([df_layer['gx'], df_layer['gy'], df_layer['gz']], axis=1)

        for neighbor_layer in layer_connections[layer]:
            if neighbor_layer not in unique_layers:
                continue
            df_neighbor = df[df['layer'] == neighbor_layer]
            coords_neighbor = cp.stack([df_neighbor['gx'], df_neighbor['gy'], df_neighbor['gz']], axis=1)

            # Use cuML NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(k_neighbours, len(df_neighbor)), algorithm='brute')
            nn.fit(coords_neighbor)
            distances, indices = nn.kneighbors(coords_layer)

            src_ids = df_layer['node_id'].to_cupy()
            dst_ids = df_neighbor['node_id'].to_cupy()[indices]

            # Flatten and append
            src_nodes.append(cp.repeat(src_ids, dst_ids.shape[1]))
            dst_nodes.append(dst_ids.reshape(-1))

    # Combine all source and destination edges
    src_nodes_all = cp.concatenate(src_nodes)
    dst_nodes_all = cp.concatenate(dst_nodes)

    edge_gdf = cudf.DataFrame({'src': src_nodes_all, 'dst': dst_nodes_all})

    # Node features
    node_gdf = df[['node_id', 'gx', 'gy', 'gz', 'layer', 'station', 'ladder', 'chip',
                   'hit_ID', 'tid', 'traj_p', 'traj_pt', 'traj_lambda', 'traj_phi']]

    # Build cuGraph Graph
    G = cugraph.Graph()
    G.from_cudf_edgelist(edge_gdf, source='src', destination='dst', renumber=True)

    return G, node_gdf


def extract_edge_features_cugraph(G, node_gdf):
    # Get edge list as DataFrame
    edge_df = G.view_edge_list()
    # Merge node features for source and destination
    merged = edge_df.merge(node_gdf, left_on='src', right_on='node_id', suffixes=('', '_src'))
    merged = merged.merge(node_gdf, left_on='dst', right_on='node_id', suffixes=('_src', '_dst'))
    # Build features as in your original code
    source_feats = merged[['gx_src', 'gy_src', 'gz_src']].to_numpy()
    target_feats = merged[['gx_dst', 'gy_dst', 'gz_dst']].to_numpy()
    feature_diff = source_feats - target_feats
    categorical_feats = merged[['layer_src', 'station_src', 'ladder_src', 'chip_src',
                                'layer_dst', 'station_dst', 'ladder_dst', 'chip_dst']].to_numpy()
    edge_features = np.concatenate([source_feats, target_feats, feature_diff, categorical_feats], axis=1)
    # Label: 1 if tid_src == tid_dst else 0
    edge_labels = (merged['tid_src'].to_numpy() == merged['tid_dst'].to_numpy()).astype(np.float32).reshape(-1, 1)
    return edge_features, edge_labels

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

# --- Main Training Loop ---

file_path = '/users/gy22186/mu3e/GNNtrain_data/train_data_snappy.parquet'
awk_data = load_data(file_path)

# Use a generator for batching
batch_gen = batch_data_generator(awk_data, frames_per_batch=1)
first_batch = next(batch_gen)
G, node_gdf = build_graph_cugraph_gpu(first_batch)
edge_feats, edge_lbls = extract_edge_features_cugraph(G, node_gdf)
model = edge_classifier(edge_feats.shape[1])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

num_epochs = 2
total_batches = len(np.unique(ak.to_numpy(awk_data['frame_array'])))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    batch_gen = batch_data_generator(awk_data, frames_per_batch=1)  # Reset generator each epoch
    pbar = tqdm(total=total_batches, desc=f"Epoch {epoch+1}", unit="batch")
    epoch_losses = []
    epoch_accuracies = []
    for batch in batch_gen:
        G, node_gdf = build_graph_cugraph(batch)
        edge_feats, edge_lbls = extract_edge_features_cugraph(G, node_gdf)
        if edge_feats.shape[0] == 0 or edge_lbls.shape[0] == 0:
            pbar.update(1)
            continue
        edge_feats_tensor = tf.convert_to_tensor(edge_feats, dtype=tf.float32)
        edge_lbls_tensor = tf.convert_to_tensor(edge_lbls, dtype=tf.float32)
        loss, acc = model.train_on_batch(edge_feats_tensor, edge_lbls_tensor)
        epoch_losses.append(loss)
        epoch_accuracies.append(acc)
        pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}")
        pbar.update(1)
    pbar.close()
    print(f"Epoch {epoch+1} - Avg Loss: {np.mean(epoch_losses):.4f}, Avg Acc: {np.mean(epoch_accuracies):.4f}")

print("Training complete! Saving model...")
model.save("trained_gnn_model_cugraph.keras")
