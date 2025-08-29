import torch
import numpy as np
import awkward as ak
import pyarrow.parquet as pq
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import networkx as nx
import pandas as pd
from torch_geometric.nn import GINEConv
import torch.nn.functional as F

# ---------- Load and Preprocess Test Data ----------
def load_data(file_path):
    table = pq.read_table(file_path)
    awk_array = ak.from_arrow(table)
    flat_dict = {key: ak.to_numpy(awk_array[key]).flatten() for key in awk_array.fields}
    return ak.Array(flat_dict)
# -------------------- Graph Builder --------------------
def build_graph_pyg(frame_hits):
    # Extract hit data
    gx = ak.to_numpy(frame_hits['gx'])
    gy = ak.to_numpy(frame_hits['gy'])
    gz = ak.to_numpy(frame_hits['gz'])
    layers = ak.to_numpy(frame_hits['layer_array'])
    stations = ak.to_numpy(frame_hits['station_array'])
    ladders = ak.to_numpy(frame_hits['ladder_array'])
    chips = ak.to_numpy(frame_hits['chip_array'])
    hit_IDs = ak.to_numpy(frame_hits['hit_ID'])
    tids = ak.to_numpy(frame_hits['tid_array'])  # Track IDs

    # Filter valid hits
    valid_mask = ~(np.isnan(gx) | np.isnan(gy) | np.isnan(gz))
    gx, gy, gz = gx[valid_mask], gy[valid_mask], gz[valid_mask]
    layers = layers[valid_mask]
    stations = stations[valid_mask]
    ladders = ladders[valid_mask]
    chips = chips[valid_mask]
    hit_IDs = hit_IDs[valid_mask]
    tids = tids[valid_mask]

    if len(gx) == 0:
        return None

    # Convert to cylindrical coordinates
    r = np.sqrt(gx**2 + gy**2)  # Radial distance
    phi = np.arctan2(gy, gx)    # Azimuthal angle
    z = gz                      # Longitudinal position

    # Normalize r and z
    r_norm = (r - r.min()) / (r.max() - r.min())
    z_norm = (z - z.min()) / (z.max() - z.min())

    # Encode phi using sin and cos
    phi_sin = np.sin(phi)
    phi_cos = np.cos(phi)

    # Combine normalized cylindrical coordinates
    coords = np.stack([r_norm, phi_sin, phi_cos, z_norm], axis=1)

    edge_index = []
    edge_attr = []
    edge_label = []

    # Group hits by layer
    layer_indices = {layer: np.where(layers == layer)[0] for layer in np.unique(layers)}

    # Define connections between adjacent layers
    layer_connections = {
        1: [2],
        2: [1, 3],
        3: [2, 4],
        4: [3]
    }

    # Create edges between all hits in adjacent layers
    for layer, neighbors in layer_connections.items():
        if layer not in layer_indices:
            continue
        for neighbor in neighbors:
            if neighbor not in layer_indices:
                continue

            source_idx = layer_indices[layer]
            target_idx = layer_indices[neighbor]

            for i in source_idx:
                for j in target_idx:
                    edge_index.append([i, j])

                    # Compute edge features
                    source = coords[i]
                    target = coords[j]
                    feat_diff = source - target
                    cat_feats = np.array([
                        layers[i], stations[i], ladders[i], chips[i],
                        layers[j], stations[j], ladders[j], chips[j]
                    ])
                    edge_feat = np.concatenate([source, target, feat_diff, cat_feats])
                    edge_attr.append(edge_feat)

                    # Label the edge as real (1) if hits belong to the same track
                    label = 1 if tids[i] == tids[j] else 0
                    edge_label.append(label)

    if not edge_index:
        return None

    # Convert edge attributes to tensors
    edge_attr = np.array(edge_attr, dtype=np.float32)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_label = torch.tensor(edge_label, dtype=torch.float).view(-1, 1)
    x = torch.tensor(coords, dtype=torch.float)

    # Add tids to the Data object
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=edge_label,
        hit_IDs=torch.tensor(hit_IDs, dtype=torch.long),
        tids=torch.tensor(tids, dtype=torch.long)  # Add tids here
    )
# ---------- Build Graph in torch_geometric Format ----------
# def build_graph_pyg(frame_hits, k_neighbours=50):
#     gx = ak.to_numpy(frame_hits['gx'])
#     gy = ak.to_numpy(frame_hits['gy'])
#     gz = ak.to_numpy(frame_hits['gz'])
#     layers = ak.to_numpy(frame_hits['layer_array'])
#     stations = ak.to_numpy(frame_hits['station_array'])
#     ladders = ak.to_numpy(frame_hits['ladder_array'])
#     chips = ak.to_numpy(frame_hits['chip_array'])
#     hit_IDs = ak.to_numpy(frame_hits['hit_ID'])
#     tids = ak.to_numpy(frame_hits['tid_array'])

#     valid_mask = ~(np.isnan(gx) | np.isnan(gy) | np.isnan(gz))
#     gx, gy, gz = gx[valid_mask], gy[valid_mask], gz[valid_mask]
#     layers = layers[valid_mask]
#     stations = stations[valid_mask]
#     ladders = ladders[valid_mask]
#     chips = chips[valid_mask]
#     hit_IDs = hit_IDs[valid_mask]
#     tids = tids[valid_mask]

#     coords = np.stack([gx, gy, gz], axis=1)
#     if coords.shape[0] == 0:
#         return None

#     edge_index = []
#     edge_attr = []
#     edge_label = []

#     layer_indices = {layer: np.where(layers == layer)[0] for layer in np.unique(layers)}
#     kdtrees = {layer: cKDTree(coords[layer_indices[layer]]) for layer in layer_indices}

#     layer_connections = {
#         1: [2],
#         2: [1, 3],
#         3: [2, 4],
#         4: [3]
#     }

#     for layer, neighbors in layer_connections.items():
#         if layer not in layer_indices:
#             continue
#         for neighbor in neighbors:
#             if neighbor not in layer_indices:
#                 continue

#             source_idx = layer_indices[layer]
#             target_idx = layer_indices[neighbor]
#             tree = kdtrees[neighbor]

#             for i in source_idx:
#                 dists, idxs = tree.query(coords[i], k=min(k_neighbours, len(target_idx)))
#                 idxs = [idxs] if np.isscalar(idxs) else idxs
#                 for j_local in idxs:
#                     j = target_idx[j_local]
#                     edge_index.append([i, j])

#                     source = coords[i]
#                     target = coords[j]
#                     feat_diff = source - target
#                     cat_feats = np.array([
#                         layers[i], stations[i], ladders[i], chips[i],
#                         layers[j], stations[j], ladders[j], chips[j]
#                     ])
#                     edge_feat = np.concatenate([source, target, feat_diff, cat_feats])
#                     edge_attr.append(edge_feat)

#                     label = 1 if tids[i] == tids[j] else 0
#                     edge_label.append(label)

#     if not edge_index:
#         return None

#     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#     edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
#     edge_label = torch.tensor(edge_label, dtype=torch.float).view(-1, 1)
#     x = torch.tensor(coords, dtype=torch.float)

#     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=edge_label, hit_IDs=torch.tensor(hit_IDs, dtype=torch.long), tids=torch.tensor(tids, dtype=torch.long))

# -------------------- Edge Classifier Model --------------------
# class EdgeClassifier(nn.Module):
#     def __init__(self, in_channels=17, hidden=[32, 64]):
#         super().__init__()
#         layers = []
#         last = in_channels
#         for i, h in enumerate(hidden):
#             layers.append(nn.Linear(last, h))
#             layers.append(nn.BatchNorm1d(h))
#             layers.append(nn.ReLU())
#             last = h
#         layers.append(nn.Linear(last, 1))  # Final output
#         self.model = nn.Sequential(*layers)

#     def forward(self, edge_attr):
#         return torch.sigmoid(self.model(edge_attr))

class GINEEdgeClassifier(nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, num_layers=3, dropout=0.3):
        super().__init__()
        # Project inputs
        self.node_proj = nn.Linear(node_in_channels, hidden_channels)
        self.edge_proj = nn.Linear(edge_in_channels, hidden_channels)

        # One distinct MLP per GINE layer 
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            nn_update = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINEConv(nn_update, edge_dim=hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.dropout = nn.Dropout(dropout)

        # Symmetric edge head: [h_src, h_dst, |h_src-h_dst|, h_src*h_dst, edge_attr]
        in_head = hidden_channels*4 + hidden_channels
        self.edge_head = nn.Sequential(
            nn.Linear(in_head, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)  # logits (no sigmoid)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr)
            x = norm(x + h)           # residual
            x = F.relu(x)
            x = self.dropout(x)

        src, dst = edge_index
        h_src, h_dst = x[src], x[dst]
        diff = torch.abs(h_src - h_dst)
        prod = h_src * h_dst
        edge_features = torch.cat([h_src, h_dst, diff, prod, edge_attr], dim=1)
        logits = self.edge_head(edge_features)
        return logits  

    
# ---------- Main Evaluation ----------
def evaluate():
    file_path = '/users/gy22186/mu3e/two_signal_files/test_data.parquet'
    print("Loading data...")
    awk_data = load_data(file_path)

    print("Building/loading graphs...")
    frame_ids = np.unique(ak.to_numpy(awk_data['frame_array']))
    graphs_path = "/users/gy22186/mu3e/two_signal_files/test_graphs/testgraphsspherical_dataset.pt"

    if os.path.exists(graphs_path):
        print("Loading saved graphs from disk...")
        data_list = torch.load(graphs_path, weights_only=False)
    else:
        print("Building graphs from raw data...")
        data_list = []
        for frame_id in tqdm(frame_ids, desc="Processing frames"):
            mask = ak.to_numpy(awk_data['frame_array']) == frame_id
            frame_hits = awk_data[mask]
            graph = build_graph_pyg(frame_hits)
            if graph is not None:
                data_list.append(graph)
        torch.save(data_list, graphs_path)
        print(f"Saved {len(data_list)} graphs to disk.")

    print(f"Total graphs ready: {len(data_list)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GINEEdgeClassifier(
        node_in_channels=4,  # Number of node features (r_norm, phi_sin, phi_cos, z_norm)
        edge_in_channels=20,  # Number of edge features
        hidden_channels=128,   # Hidden layer size
        num_layers=3          # Number of GINEConv layers
    ).to(device)
    model.load_state_dict(torch.load("/users/gy22186/mu3e/two_signal_files/edge_classifier_gine.pt", map_location=device))
    model.eval()

    loader = DataLoader(data_list, batch_size=1, shuffle=False)
    # for batch in loader:
    #     print(f"Batch attributes: {batch.keys}")  # Check all attributes in the batch
    #     break

    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Running Inference"):
            batch = batch.to(device)
            # Pass x, edge_index, and edge_attr to the model
            pred = model(batch.x, batch.edge_index, batch.edge_attr).view(-1)
            scores = pred.cpu().numpy()
            edges = batch.edge_index.cpu().numpy().T
            predictions.append({"edges": edges, "scores": scores})

    return predictions, data_list


def build_tracks(
    predictions, 
    data_list, 
    threshold_lo=0.9,  
    min_hits=4, 
    output_path="/users/gy22186/mu3e/two_signal_files/reconstructed_tracks_new.csv"
):
    all_tracks = []

    for batch_idx, (pred_dict, batch) in enumerate(zip(predictions, data_list)):
        edges = np.array(pred_dict["edges"])
        scores = np.array(pred_dict["scores"])

        # Filter edges above the threshold
        mask = scores > threshold_lo
        high_score_edges = edges[mask]
        high_score_values = scores[mask]

        G = nx.Graph()
        for (i, j), s in zip(high_score_edges, high_score_values):
            G.add_edge(i, j, weight=s)

        components = list(nx.connected_components(G))

        hit_IDs = batch.hit_IDs.cpu().numpy()
        tids = batch.tids.cpu().numpy()

        for track_id, component in enumerate(components):
            if len(component) < min_hits:
                continue

            component = list(component)
            hit_id_list = hit_IDs[component].tolist()
            tid_list = tids[component].tolist()

            all_tracks.append({
                "batch": batch_idx,
                "track_id": track_id,
                "hit_IDs": hit_id_list,
                "tids": tid_list,
                "num_hits": len(component)
            })

    df = pd.DataFrame(all_tracks)
    df.to_csv(output_path, index=False)
    print(f"Saved reconstructed tracks to '{output_path}'")


if __name__ == "__main__":
    preds, data = evaluate()
    build_tracks(preds, data)
