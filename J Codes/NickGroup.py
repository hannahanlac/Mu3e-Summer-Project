import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import os
import re


output_dir = 'ProcessedData/signal1_96_32652/evaluation_prep'
transformer_predictions = pd.read_parquet('ProcessedData/signal1_96_32652/test_data/test_data.parquet')

merged_hits_truth_data_filtered_columns = transformer_predictions[[
"frame_array","tid_array", "traj_p", "traj_pt", "traj_lambda", "traj_phi"]].copy()
merged_hits_truth_data_filtered_columns['num_hits'] = merged_hits_truth_data_filtered_columns.groupby('tid_array')['tid_array'].transform('count')
unique_reconstructable_tid_arrays = merged_hits_truth_data_filtered_columns[merged_hits_truth_data_filtered_columns['num_hits'] >= 4].drop_duplicates(subset=['tid_array']) 


predicted_tracks = pd.read_csv('ProcessedData/signal1_96_32652/reconstructed_tracks/predicted_tracks3.csv')

# Drop rows where correct_track is 0
reconstructed_tracks = predicted_tracks[predicted_tracks['correct_track'] != 0]

reconstructed_tracks['tids'] = reconstructed_tracks['tids'].apply(lambda x: int(re.sub(r'\D', '', x)))

reconstructed_tracks = reconstructed_tracks.rename(columns={'tids': 'tid'})


# Merge unique tids with predicted tracks
merged_tracks = unique_reconstructable_tid_arrays.merge(
    reconstructed_tracks,
    left_on='tid_array',      
    right_on='tid',              
    how='left'
    )


unique_reconstructable_tid_arrays_path = os.path.join(output_dir, "NickGroup2.csv")
merged_tracks.to_csv(unique_reconstructable_tid_arrays_path, index = False)


#merged_tracks.drop(columns=['frameNumber_y'], inplace=True)


