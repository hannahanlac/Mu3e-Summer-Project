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

def load_data(file_path):
    table = pq.read_table(file_path)  # Load Parquet file as an Arrow table
    return ak.from_arrow(table)       # Convert Arrow table to an Awkward Array


def batch_data(awk_array, frame_size=10):
    num_hits = len(awk_array['hit_ID'])
    return [awk_array[i:i + frame_size] for i in range(0, num_hits, frame_size)]

def standardize_features(data):
    scaler = StandardScaler()
    
    # Convert Awkward Array to a NumPy array (ensuring it's 2D)
    gx = ak.to_numpy(data['gx'])
    gy = ak.to_numpy(data['gy'])
    gz = ak.to_numpy(data['gz'])
    
    stacked_features = np.column_stack([gx, gy, gz])  # Ensures shape (num_samples, 3)
    
    return scaler.fit_transform(stacked_features)


class EdgeClassifier(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.dense1 = Dense(hidden_dim, activation='relu')
        self.batch_norm = BatchNormalization()
        self.dense2 = Dense(hidden_dim, activation='relu')
        self.output_layer = Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.batch_norm(x)
        x = self.dense2(x)
        return self.output_layer(x)

def train_model(x_train, y_train, model, epochs=10, batch_size=32):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

def cluster_hits(features, predictions, threshold=0.5):
    print("Features shape:", features.shape)
    print("Predictions shape:", predictions.shape)
    print("Threshold mask shape:", (predictions > threshold).shape)

    mask = (predictions > threshold).squeeze()  # Convert (771480, 1) -> (771480,)

    print("Features shape:", features.shape)
    print("Predictions shape:", predictions.shape)
    print("Threshold mask shape:", (predictions > threshold).shape)

    clustering = DBSCAN(eps=1.5, min_samples=2).fit(features[mask])
    return clustering.labels_


file_path = 'ProcessedData/signal1_96_32652/train_data/train_data.parquet'
awk_data = load_data(file_path)
batches = batch_data(awk_data)

features = np.vstack([standardize_features(batch) for batch in batches])
labels = np.concatenate([np.ones(len(batch)) for batch in batches])  # Placeholder labels
features, labels = shuffle(features, labels)

model = EdgeClassifier(input_dim=3, hidden_dim=16)
train_model(features, labels, model)

model.save('track_finding_gnn')

predictions = model.predict(features)
all_clusters = []
for i, batch in enumerate(batches):
    batch_features = standardize_features(batch)
    batch_predictions = model.predict(batch_features)
    
    # Perform clustering on the current batch
    batch_clusters = cluster_hits(batch_features, batch_predictions)
    
    all_clusters.append(batch_clusters)

# Convert list of cluster labels into a single array
final_clusters = np.concatenate(all_clusters)

print(f"Cluster Assignments: {all_clusters}")

plt.scatter(features[:, 0], features[:, 1], c=all_clusters, cmap='viridis')
plt.xlabel('gx')
plt.ylabel('gy')
plt.title('Hit Clustering')
plt.colorbar()
plt.show()
