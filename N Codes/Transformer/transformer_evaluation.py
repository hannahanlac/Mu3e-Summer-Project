import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import os

def EvaluateTransformerTracks(transformer_predictions, output_dir, truth_condition):
    """Function that takes a set of output predictions of the transformer algorithm,
    and groups them into tracks, and evaluates how good these tracks are based on the metric provided.
    Inputs:
    - transformer prediction csv
    - Output directory for saving
    - Truth condition using: Can be either 'current_alg' for direct comparison, '50/50','75/75' or '100/100'
    Outputs:
    - EFFICIENCY evaluating csv
    - FAKE RATE evaluating csv """
    
    assert truth_condition in ['current_alg', '50/50', '75/75', '100/100'], "fake_measure must be 'current_alg', '50/50', '75/75' or '100/100'"

    # Sort the hits data to find out how many unique tids there are (i.e how many true particles). Use this to finish truth condition on tracks later
    merged_hits_truth_data_filtered_columns = transformer_predictions[[
    "frameNumber","tid", "traj_p", "traj_pt", "traj_lambda", "traj_phi", "bin_index"]].copy()
    merged_hits_truth_data_filtered_columns['num_hits'] = merged_hits_truth_data_filtered_columns.groupby('tid')['tid'].transform('count')
    unique_reconstructable_tids = merged_hits_truth_data_filtered_columns[merged_hits_truth_data_filtered_columns['num_hits'] >= 4].drop_duplicates(subset=['tid']) 
    unique_reconstructable_tids_path = os.path.join(output_dir, "reconstructable_tids.csv")
    unique_reconstructable_tids.to_csv(unique_reconstructable_tids_path, index = False)


    #Group the transformer predictions and evaluate the tracks based on the truth condtition - First half of the 50/50 etc rule
    transformer_predictions_grouped = transformer_predictions.groupby(['frameNumber','predicted_bin_index']).apply(lambda group: TrackTruthSorting(group, truth_condition))
    #Filter the tracks down
    # grouped_predictions_path = os.path.join(output_dir, "grouped_predictions_signal1_96.csv")
    # transformer_predictions_grouped.to_csv(grouped_predictions_path, index=False)

    tracks_only = transformer_predictions_grouped.drop(columns=['hitIndex', 'tid','traj_p','traj_pt','traj_lambda','traj_phi','bin_index'])  # Drop unnecessary columns if they exist
    tracks_only = (tracks_only
                   .drop_duplicates(subset=['frameNumber','predicted_bin_index'])
                   .reset_index(drop=True))

    ####### EFFICIENCY CALCULATIONS HERE ##########
    correct_tracks = tracks_only[tracks_only['true_track'] == 1]

    # Merge on 'tid' from unique_reconstructable_tids and 'true_track_tid' from tracks_only
    merged_tracks = unique_reconstructable_tids.merge(
        correct_tracks,
        left_on='tid',      
        right_on='true_track_tid',              
        how='left')
    merged_tracks.drop(columns=['frameNumber_y'], inplace=True)

    # Check 2nd part of the 50/50 etc conditions: Do the true tracks hold >50% of the tid hits? If not discard
    if truth_condition == '50/50':
        merged_tracks['true_track'] = merged_tracks.apply(
            lambda row: 0 if row['num_tid_hits_in_track'] < 0.5 * row['num_hits'] else row['true_track'],
            axis=1)
    elif truth_condition == '75/75':
        merged_tracks['true_track'] = merged_tracks.apply(
            lambda row: 0 if row['num_tid_hits_in_track'] < 0.75 * row['num_hits'] else row['true_track'],
            axis=1)
    elif truth_condition == '100/100':
        merged_tracks['true_track'] = merged_tracks.apply(
            lambda row: 0 if row['num_tid_hits_in_track'] <  row['num_hits'] else row['true_track'],
            axis=1)
         
    # Save merged result - This is csv of all reconstructable tracks, and the correct reconstructed ones.
    merged_tracks_path = os.path.join(output_dir, "merged_tracks.csv") 
    merged_tracks.to_csv(merged_tracks_path, index=False)

    # Update the tracks_only dataframe with the true_track values from the merge abovve. This now satisfies fully the 50/50 etc condition
    mapping = (
        merged_tracks[['true_track_tid', 'true_track']]
        .drop_duplicates(subset='true_track_tid')
        .set_index('true_track_tid')['true_track']
    )
    tracks_only['true_track'] = tracks_only['true_track_tid'].map(mapping).fillna(tracks_only['true_track']).astype(int)
    tracks_only_correct_path = os.path.join(output_dir, "tracks_only_updated.csv")
    tracks_only.to_csv(tracks_only_correct_path, index=False)

    # Calculate overall efficiency measure
    num_reconstructable_tracks = len(merged_tracks)
    num_correct_tracks = (merged_tracks['true_track'] ==1).sum()
    print(f"num_correct_tracks:{num_correct_tracks}")
    overall_eff = (num_correct_tracks / num_reconstructable_tracks) *100
    print(f"overall efficiency of transformer:{overall_eff}")


    # Calculate overall Fake Rate
    num_reconstructed_tracks = len(tracks_only)
    num_fake_tracks = (tracks_only['true_track']==0).sum()
    print(f"Total number of constructed tracks:{num_reconstructed_tracks}")
    print(f"Number of Fake Tracks:{num_fake_tracks}")
    overall_fake_rate = (num_fake_tracks/num_reconstructed_tracks) *100
    print(f"Overall Fake Rate:{overall_fake_rate}")


    return merged_tracks, tracks_only


def TrackTruthSorting(group, truth_condition):
        # Check if all predicted_bin_index are equal to bin_index
    
    match_ratio = (group['predicted_bin_index'] == group['bin_index']).mean()

    if truth_condition == 'current_alg':
        true_track = match_ratio == 1

    elif truth_condition == '50/50':
        true_track = match_ratio > 0.5

    elif truth_condition == '75/75':
        true_track = match_ratio > 0.75

    elif truth_condition == '100/100':
        true_track = match_ratio == 1

    if true_track:
            group['true_track'] = 1
            true_track_tid = group['tid'].mode()[0]  # Most common tid
            group['true_track_tid'] = true_track_tid  # Assign the most common tid to all rows
            group['num_tid_hits_in_track'] = (group['tid'] == group['true_track_tid']).sum()  
    else:
            group['true_track'] = 0
            #group['tids_in_track'] = ', '.join(map(str, group['tid'].unique())) #Not sure I actually need this

    group['num_hits_in_track'] = group['hitIndex'].nunique()
    return group




def EfficiencyLambdaMomentumPlot(data_file, min_lam, max_lam, lam_res, min_p, max_p, p_res, p_type, num_hits):
    """Function that produces a plot of track finding algorithm efficiency as a function of total momentum
       and inclination angle (lambda) of the underlying truth particle. Creates a 2D colourmap of efficiency as output.
    Input:
    - Data file containing:
        - No. unique particle IDs registering 4 hits (1 per layer... supposedly)
        - Truth information for unique particle IDs (mommentum and lambda)
        - No. of reconstructed tracks. 
    - Lambda and Momentum ranges, and resolutions
    - p_type: traj_p (total p) or traj_pt (transverse)
    - num_hits: Min num hits used (4 by default for all 'reconstructable' tracks)
    - truth_measure: 'absolute' for strongest, otherwise all reconstructed tracks considered
    Output:
    - Colourmap of efficiency (normalised) as function of truth momentum and lambda of the particle IDs
    """
    print("Data loaded successfully for efficiencyMomentumPlot")

    # data_file = data_file.copy()
    # data_file['abs_true_track'] = ((data_file['mc_prime'] == 1) & (data_file['mc measure'] == 1)).astype(int)
    # data_file = data_file.sort_values(by='abs_true_track', ascending=False)
    # data_file = data_file.drop_duplicates(subset='tid', keep="first") 
    data_file = data_file[data_file['num_hits'] >= num_hits]  # Allows control here of if want to do for just longer tracks. Leave at 4 for no change

    print("no.unique tids:", data_file['tid'].nunique())
    print("no.entries:",len(data_file))

    # Bins
    lambda_bins = np.arange(min_lam, max_lam + lam_res, lam_res)
    momentum_bins = np.arange(min_p, max_p + p_res, p_res)

    #Bin indices
    data_file['lambda_bin'] = np.digitize(data_file['traj_lambda'], lambda_bins) - 1 #-1 to ensure from 0 indexing
    data_file['momentum_bin'] = np.digitize(data_file[p_type], momentum_bins) - 1
    data_file['lambda_bin'] = np.clip(data_file['lambda_bin'], 0, len(lambda_bins) - 2)
    data_file['momentum_bin'] = np.clip(data_file['momentum_bin'], 0, len(momentum_bins) - 2)

    # All possible bin combinations indexed. NOTE: This is needed to ensure no bins are dropped later! (This was found out the hard way...)
    bin_index = pd.MultiIndex.from_product(
        [range(len(lambda_bins) - 1), range(len(momentum_bins) - 1)],
        names=[ 'lambda_bin','momentum_bin']
    )


    grouped = data_file.groupby([ 'lambda_bin','momentum_bin']).agg(
        total_tracks=('tid', 'count'),
        true_tracks=('true_track', 'sum')
    ).reindex(bin_index, fill_value=0)  # Ensure no bins dropped


    # Efficiency and efficiency matrix of correct dimensions
    grouped['efficiency'] = grouped['true_tracks'] / grouped['total_tracks'].replace(0, np.nan)
    efficiency_matrix = grouped['efficiency'].unstack(fill_value=0)

    if p_type == 'traj_pt':
        p_plot_name = 'pt'
    elif p_type == 'traj_p':
        p_plot_name = 'p'
    
    #Plot
    plt.figure(figsize=(10, 6))
        
    levels = np.arange(0, 1.1, 0.1) # Use for controlling colourmap no. colours
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=256)
    mesh = plt.pcolormesh(lambda_bins, momentum_bins, efficiency_matrix.T, cmap='viridis', norm=norm, shading='auto')
    cbar = plt.colorbar(mesh, ticks=levels, pad=0.0)
    cbar.set_label('Efficiency', fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=16, length=8, width=2) 
    ax.tick_params(axis='both', which='minor', labelsize=12, length=4, width=1)  
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    plt.xlabel("$\lambda$ [rad]", fontsize=18)
    plt.ylabel(f"{p_plot_name} [MeV/c]", fontsize=18)
    plt.title(f"Efficiency as a Function of $\lambda$ and {p_plot_name}", fontsize=20)


    plt.show()
    return




transformer_prediction_file = "/root/Mu3eProject/DataFilesAndTests/TransformerData/TestForEffPlots/predictions_test_with_truths_signal1_96.csv"
output_dir = "/root/Mu3eProject/DataFilesAndTests/TransformerData/TestForEffPlots/"


transformer_predictions = pd.read_csv(transformer_prediction_file)
#merged_tracks = EvaluateTransformerTracks(transformer_predictions, output_dir, truth_condition = 'current_alg')
merged_tracks = EvaluateTransformerTracks(transformer_predictions, output_dir, truth_condition = '50/50')[0]
#merged_tracks = EvaluateTransformerTracks(transformer_predictions, output_dir, truth_condition = '75/75')
#merged_tracks = EvaluateTransformerTracks(transformer_predictions, output_dir, truth_condition = '100/100')

EfficiencyLambdaMomentumPlot(merged_tracks, min_lam = -1.6,max_lam = 1.6, lam_res = 0.05 , min_p = 0, max_p = 60, p_res = 1, p_type = 'traj_pt', num_hits = 4)
EfficiencyLambdaMomentumPlot(merged_tracks, min_lam = -1.6,max_lam = 1.6, lam_res = 0.05 , min_p = 0, max_p = 60, p_res = 1, p_type = 'traj_p', num_hits = 4)



