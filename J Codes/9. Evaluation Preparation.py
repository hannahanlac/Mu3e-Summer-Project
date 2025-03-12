import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import os
import ast

# Test Data
output_dir = 'ProcessedData/signal1_96_32652/evaluation_prep'

test_data = pd.read_parquet('ProcessedData/signal1_96_32652/test_data/test_data.parquet')

merged_hits_truth_data_filtered_columns = test_data[[
"frame_array","tid_array", "traj_p", "traj_pt", "traj_lambda", "traj_phi"]].copy()

merged_hits_truth_data_filtered_columns['num_hits_x'] = merged_hits_truth_data_filtered_columns.groupby('tid_array')['tid_array'].transform('count')
unique_reconstructable_tid_arrays = merged_hits_truth_data_filtered_columns[merged_hits_truth_data_filtered_columns['num_hits_x'] >= 4].drop_duplicates(subset=['tid_array']) 

# predicted Data
predicted_tracks = pd.read_csv('ProcessedData/signal1_96_32652/reconstructed_tracks/predicted_tracks3.csv')

# Drop rows where correct_track is 0
reconstructed_tracks = predicted_tracks[predicted_tracks['correct_track'] != 0]

reconstructed_tracks["tid"] = reconstructed_tracks["tids"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x[0])

# Ensure hit_IDs is treated as a list
reconstructed_tracks["hit_IDs"] = reconstructed_tracks["hit_IDs"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

reconstructed_tracks = reconstructed_tracks[["tid", "num_hits", "hit_IDs", "correct_track"]]



# Merge unique tids with predicted tracks
merged_tracks = unique_reconstructable_tid_arrays.merge(
    reconstructed_tracks,
    left_on='tid_array',      
    right_on='tid',              
    how='left'
    )

merged_tracks["hit_IDs"] = merged_tracks["hit_IDs"].apply(lambda x: x if isinstance(x, list) else [])

# **Cluster hit_IDs per tid (combine and remove duplicates)**
grouped_tracks = merged_tracks.groupby("tid_array").agg(
    traj_p=("traj_p", "first"), 
    traj_pt=("traj_pt", "first"), 
    traj_lambda=("traj_lambda", "first"), 
    traj_phi=("traj_phi", "first"), 
    num_hits_x=("num_hits_x", "first"),  # Keep original num_hits from test data
    hit_IDs=("hit_IDs", lambda x: list(set(sum(x.dropna(), [])))),  # Merge lists, remove NaN, and deduplicate
)

# Count unique hits per tid
grouped_tracks["num_hits_y"] = grouped_tracks["hit_IDs"].apply(len)

grouped_tracks["true_track"] = grouped_tracks["hit_IDs"].apply(lambda x: 1 if len(x) > 0 else 0)

# Reset index
grouped_tracks.reset_index(inplace=True)

final_output_path = os.path.join(output_dir, "NickGroup6.csv")
grouped_tracks.to_csv(final_output_path, index=False)

print(f"Clustered tracks saved to {final_output_path}")


