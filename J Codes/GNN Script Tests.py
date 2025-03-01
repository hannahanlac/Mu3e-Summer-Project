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

