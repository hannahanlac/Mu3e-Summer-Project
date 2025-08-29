import os
import numpy as np
import awkward as ak
import pyarrow.parquet as pq
from tqdm import tqdm
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.multiprocessing as mp
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch_geometric.nn import GINEConv
import torch.nn.functional as F


# -------------------- Data Loading --------------------
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
    hit_IDs = ak.to_numpy(frame_hits['hit_ID'])
    tids = ak.to_numpy(frame_hits['tid_array'])  # Track IDs

    # Filter valid hits
    valid_mask = ~(np.isnan(gx) | np.isnan(gy) | np.isnan(gz))
    gx, gy, gz = gx[valid_mask], gy[valid_mask], gz[valid_mask]
    layers = layers[valid_mask]
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

    # Combine normalised cylindrical coordinates
    coords = np.stack([r_norm, phi_sin, phi_cos, z_norm], axis=1)

    edge_index = []
    edge_attr = []
    edge_label = []

    # One-hot encode the layers with a fixed number of categories
    encoder = OneHotEncoder(sparse_output=False, categories=[range(1, 5)])  # Fixed categories: [1, 2, 3, 4]
    layers_onehot = encoder.fit_transform(layers.reshape(-1, 1))  # One-hot encode layers

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
                    layer_onehot_source = layers_onehot[i]
                    layer_onehot_target = layers_onehot[j]
                    edge_feat = np.concatenate([source, target, feat_diff, layer_onehot_source, layer_onehot_target])
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
    x = torch.tensor(coords, dtype=torch.float)  # Node features remain in cylindrical coordinates

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=edge_label, hit_IDs=torch.tensor(hit_IDs, dtype=torch.long))


def evaluate_edge_coverage(graph: Data, tids: np.ndarray, layers: np.ndarray):
    edge_index = graph.edge_index.cpu().numpy()
    predicted_edges = set(tuple(edge) for edge in edge_index.T if edge[0] != edge[1])

    # All ground truth edges: same-tid pairs in adjacent layers
    true_edges = set()
    fake_edges = set()
    tid_to_hits = {}

    for idx, tid in enumerate(tids):
        if tid < 0:
            continue  # skip noise
        tid_to_hits.setdefault(tid, []).append(idx)

    for hit_list in tid_to_hits.values():
        for i in range(len(hit_list)):
            for j in range(i + 1, len(hit_list)):
                hit_i = hit_list[i]
                hit_j = hit_list[j]
                # Only include edges between hits in adjacent layers
                if abs(layers[hit_i] - layers[hit_j]) == 1:
                    true_edges.add((hit_i, hit_j))
                    true_edges.add((hit_j, hit_i))  # bidirectional

    # Identify all possible fake edges (different-tid pairs in adjacent layers)
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            if tids[i] != tids[j] and abs(layers[i] - layers[j]) == 1:
                fake_edges.add((i, j))
                fake_edges.add((j, i))  # bidirectional

    if len(true_edges) == 0:
        print("No true edges to evaluate.")
        return {"total": 0, "found": 0, "missed": 0, "coverage": 0.0}

    # Evaluate true edges
    found_true = true_edges & predicted_edges
    missed_true = true_edges - predicted_edges

    # Evaluate fake edges
    found_fake = fake_edges & predicted_edges
    missed_fake = fake_edges - predicted_edges

    # Compute coverage metrics
    true_coverage = len(found_true) / len(true_edges) if len(true_edges) > 0 else 0.0
    fake_coverage = len(found_fake) / len(fake_edges) if len(fake_edges) > 0 else 0.0

    print(f"Total true edges (same-tid pairs in adjacent layers): {len(true_edges)}")
    print(f"Found true edges: {len(found_true)}")
    print(f"Missing true edges: {len(missed_true)}")
    print(f"True edge coverage: {true_coverage:.2%}")

    print(f"Total fake edges (different-tid pairs in adjacent layers): {len(fake_edges)}")
    print(f"Found fake edges: {len(found_fake)}")
    print(f"Missing fake edges: {len(missed_fake)}")
    print(f"Fake edge coverage: {fake_coverage:.2%}")

    return {
        "total_true_edges": len(true_edges),
        "found_true_edges": len(found_true),
        "missed_true_edges": len(missed_true),
        "true_coverage": true_coverage,
        "total_fake_edges": len(fake_edges),
        "found_fake_edges": len(found_fake),
        "missed_fake_edges": len(missed_fake),
        "fake_coverage": fake_coverage
    }

# -------------------- Edge Classifier Model --------------------
# class EdgeClassifier(nn.Module):
#     def __init__(self, in_channels=20, hidden=[32, 64]):
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
        return logits  # use with BCEWithLogitsLoss

# -------------------- Training Function --------------------
def train(model, loader, optimizer, criterion, device, max_grad_norm=1.0):
    model.train()
    epoch_losses, epoch_accuracies = [], []
    pbar = tqdm(loader, desc="Training", unit="batch")

    for batch in pbar:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr).view(-1)
        label = batch.y.view(-1)

        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)  
            acc = ((probs > 0.5) == (label > 0.5)).float().mean().item()

        epoch_losses.append(loss.item())
        epoch_accuracies.append(acc)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    return sum(epoch_losses)/len(epoch_losses), sum(epoch_accuracies)/len(epoch_accuracies)

# -------------------- Main --------------------
def main():
    file_path = '/users/gy22186/mu3e/two_signal_files/train_data.parquet'
    print("Loading data...")
    awk_data = load_data(file_path)

    print("Building/loading graphs...")
    frame_ids = np.unique(ak.to_numpy(awk_data['frame_array']))
    graphs_path = "/users/gy22186/mu3e/two_signal_files/train_graphs/graphslayersonly_dataset.pt"

    if os.path.exists(graphs_path):
        print("Loading saved graphs from disk...")
        data_list = torch.load(graphs_path, weights_only=False)
    else:
        print("Building graphs...")
        data_list = []
        for frame_id in tqdm(frame_ids, desc="Processing frames"):
            mask = ak.to_numpy(awk_data['frame_array']) == frame_id
            frame_hits = awk_data[mask]
            graph = build_graph_pyg(frame_hits)
            if graph is not None:
                data_list.append(graph)
                # tids = ak.to_numpy(frame_hits['tid_array'])[~np.isnan(ak.to_numpy(frame_hits['gx']))]  
                # layers = ak.to_numpy(frame_hits['layer_array'])[~np.isnan(ak.to_numpy(frame_hits['gx']))]  
                # evaluate_edge_coverage(graph, tids, layers) # uncomment these 3 lines if you want to evaluate the proportion of real/fake edges included in the graph
        torch.save(data_list, graphs_path)
        print(f"Saved {len(data_list)} graphs to disk.")

    print(f"Total graphs ready: {len(data_list)}")
    # for i, graph in enumerate(data_list): # uncomment for edge_attribute shape debugging
    #     print(f"Graph {i}: edge_attr.shape = {graph.edge_attr.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GINEEdgeClassifier(
        node_in_channels=4,  # Number of node features (r_norm, phi_sin, phi_cos, z_norm)
        edge_in_channels=20,  # Number of edge features
        hidden_channels=128,   # Hidden layer size
        num_layers=3          # Number of GINEConv layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()  

    loader = DataLoader(data_list, batch_size=100, shuffle=True)

    print("Starting training...")
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        avg_loss, avg_acc = train(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch:02d} Summary | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f}")

        scheduler.step(avg_loss)
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

    torch.save(model.state_dict(), "/users/gy22186/mu3e/two_signal_files/edge_classifier_gine.pt")
    print("Training complete. Model saved as edge_classifier_gine.pt")

if __name__ == "__main__":
    main()


