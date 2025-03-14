import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import os
import ast

# Import test data file
test_data = pd.read_parquet('ProcessedData/signal1_96_32652/test_data/test_data.parquet')

# Specify which data columns want to keep
test_data_columns = test_data[[
"frame_array","tid_array", "traj_p", "traj_pt", "traj_lambda", "traj_phi"]].copy()

# Count number of hit_IDs in each unique tid, drop any tracks with num hits < 4 (as not reconstructable - no use (unphysical/detector error))
test_data_columns['num_hits_x'] = test_data_columns.groupby('tid_array')['tid_array'].transform('count')
unique_reconstructable_tids = test_data_columns[test_data_columns['num_hits_x'] >= 4].drop_duplicates(subset=['tid_array']) 

###################################################

# Import predicted tracks from GNN
predicted_tracks = pd.read_csv('ProcessedData/signal1_96_32652/reconstructed_tracks/predicted_tracks3.csv')


def TrackTruthSorting(group, truth_condition):
    """Function that filters the predicted tracks based on the truth condition given. Either:
        - Current_alg: All of the hits in the track must belong to the same particle
        - 50/50: At least 50% hits from the same particle, at least 50% of particle hits in track
        - 75/75 or 100/100: As for 50/50 but with increased percentages - stronger measures"""
    
    match_ratio = (group['predicted_bin_index'] == group['bin_index']).mean()
    if truth_condition == 'current_alg':
        correct_track = match_ratio == 1

    elif truth_condition == '50/50':
        correct_track = match_ratio > 0.5

    elif truth_condition == '75/75':
        correct_track = match_ratio > 0.75

    elif truth_condition == '100/100':
        correct_track = match_ratio == 1

    if correct_track:
            group['correct_track'] = 1
            correct_track_tid = group['tid'].mode()[0]  # Finding the most common tid in the track - this is the tid the track represents
            group['correct_track_tid'] = correct_track_tid 
            group['num_tid_hits_in_track'] = (group['tid'] == group['correct_track_tid']).sum()  
    else:
            group['correct_track'] = 0
            #group['tids_in_track'] = ', '.join(map(str, group['tid'].unique())) #Not sure I actually need this

    group['num_hits_in_track'] = group['hitIndex'].nunique()
    return group



