import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import os
import ast

def EfficiencyLambdaMomentumPlot(data_file, min_lam, max_lam, lam_res, min_p, max_p, p_res, p_type, num_hits_x):
    """Function that produces a plot of track finding algorithm efficiency as a function of total momentum
       and inclination angle (lambda) of the underlying truth particle. Creates a 2D colourmap of efficiency as output.
    Input:
    - Data file containing:
        - No. unique particle IDs registering 4 hits (1 per layer... supposedly)
        - Truth information for unique particle IDs (mommentum and lambda)
        - No. of reconstructed tracks. 
    - Lambda and Momentum ranges, and resolutions
    - p_type: traj_p (total p) or traj_pt (transverse)
    - num_hits_x: Min num hits used (4 by default for all 'reconstructable' tracks)
    - truth_measure: 'absolute' for strongest, otherwise all reconstructed tracks considered
    Output:
    - Colourmap of efficiency (normalised) as function of truth momentum and lambda of the particle IDs
    """
    print("Data loaded successfully for efficiencyMomentumPlot")

    data_file = data_file[data_file['num_hits_x'] >= num_hits_x]  # Allows control here of if want to do for just longer tracks. Leave at 4 for no change

    # print("no.unique tids:", data_file['tid'].nunique())
    # print("no.entries:",len(data_file)) #Checking match, if not something wrong

    # Bins
    lambda_bins = np.arange(min_lam, max_lam + lam_res, lam_res)
    momentum_bins = np.arange(min_p, max_p + p_res, p_res)
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
        correct_tracks=('correct_track', 'sum')
    ).reindex(bin_index, fill_value=0)  # Ensure no bins dropped!!


    # Efficiency and efficiency matrix of correct dimensions
    grouped['efficiency'] = grouped['correct_tracks'] / grouped['total_tracks'].replace(0, np.nan)
    efficiency_matrix = grouped['efficiency'].unstack(fill_value=0)

    p_plot_name = 'p' if p_type == 'traj_p' else 'pt'
    
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
    #plt.title(f"Efficiency as a Function of $\lambda$ and {p_plot_name}", fontsize=20)

    plt.show()
    return

def EfficiencyTrackLengthPlot(data_file):
    """Function to plot the efficiency of the track finding as a function of the number of registered hits. Can be done for variable truth definitions
    Input:
    - Data file of transformer tracks and if they are true/not
    Output:
    - Overall efficiency measure
    """

    print("Processing Efficiency Track Length Plot...")

    # Ensure required columns exist
    required_columns = {'tid', 'num_hits_x', 'correct_track'}
    missing_columns = required_columns - set(data_file.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in data file: {missing_columns}")
 
    bins = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, float('inf')]  # 10+ is grouped as the last bin
    bin_labels = ['4', '5', '6', '7', '8', '9','10', '11', '12', '12+']
    data_file['hit_bin'] = pd.cut(data_file['num_hits_x'], bins=bins, labels=bin_labels, right=True)

    # Group data by 'hit_bin' and calculate efficiency for each bin
    grouped_data_tracks = data_file.groupby('hit_bin', observed=False)
    total_tracks_per_length = grouped_data_tracks['tid'].count()
    reconstructed_tracks_per_hit = grouped_data_tracks['correct_track'].sum() # Total tracks per group - Only counts where a reconstruction was made.
    
    efficiency_all = reconstructed_tracks_per_hit / total_tracks_per_length #Efficiency for all recon tracks
    print(efficiency_all)

    plt.figure(figsize=(10, 6))
    plt.bar(efficiency_all.index, efficiency_all.values, color='blue', edgecolor = 'black', label='Efficiency')
    plt.xlabel("Track Length", fontsize=14)
    plt.ylabel("Efficiency", fontsize=14)
    #plt.title("Efficiency as a Function of Track Length", fontsize=18)
    plt.grid(True)
    plt.legend()
    plt.yticks(np.arange(0, 1.1, 0.1))  
    plt.show()
    return

def FakeRateTrackLengthPlot(data_file):
    """Function to plot the fake rate of the track finding as a function of the reconstructed track length. 
    Input:
    - Data file of all tracks reconstructed, fake and real
    Output:
    - Fake rate measure
    """

    # Ensure required columns exist
    required_columns = {'num_hits_x', 'correct_track'}
    missing_columns = required_columns - set(data_file.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in data file: {missing_columns}")

    # Bins for track length
    bins = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, float('inf')]
    bin_labels = ['4', '5', '6', '7', '8', '9', '10', '11', '12', '12+']
    data_file['track_length_bin'] = pd.cut(data_file['num_hits_x'], bins=bins, labels=bin_labels, right=True)

    # Debugging: Check bin assignments
    print("Track length bin counts:\n", data_file['track_length_bin'].value_counts())

    grouped_data_tracklength = data_file.groupby('track_length_bin')

    total_tracks_per_length = grouped_data_tracklength['correct_track'].count()  # Total tracks per group
    fake_tracks = grouped_data_tracklength['correct_track'].agg(lambda x: (x == 0).sum())
    fake_rate = fake_tracks / total_tracks_per_length

    # Replace NaN values in fake_rate with zeros
    fake_rate = fake_rate.fillna(0)

    # Debugging: Check fake rate values
    print("Fake rate by track length:\n", fake_rate)

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(fake_rate.index, fake_rate.values, color='red', edgecolor='black', alpha=0.8)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.3f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xlabel("Track Length", fontsize=14)
    plt.ylabel("Fake Rate", fontsize=14)
    #plt.title("Fake Rate vs Track Length", fontsize=16)
    plt.xticks(fake_rate.index)  
    plt.ylim(0, max(fake_rate.values) * 1.2) 
    plt.show()

    return fake_rate


def calculate_fake_rate(predicted_tracks_file):
    """Calculate the fake rate from the predicted tracks CSV file."""
    predicted_tracks = pd.read_csv(predicted_tracks_file)
    
    # Count tracks where correct_track = 0
    fake_tracks = (predicted_tracks["correct_track"] == 0).sum()
    
    # Total number of tracks
    total_tracks = len(predicted_tracks)
    
    # Compute fake rate
    fake_rate = fake_tracks / total_tracks if total_tracks > 0 else 0
    print(f"\n Fake Rate: {fake_rate:.4f} ({fake_tracks}/{total_tracks})")

    return fake_rate


# Define a safe parsing function for the 'hit_IDs' column
def safe_parse_hit_ids(value):
    try:
        # If the value is a string, evaluate it as a Python literal
        if isinstance(value, str):
            return ast.literal_eval(value)
        # If the value is a NumPy array, convert it to a list
        elif isinstance(value, np.ndarray):
            return value.tolist()
        # If the value is already a list, return it as-is
        elif isinstance(value, list):
            return value
        # For any other type, return an empty list
        else:
            return []
    except (ValueError, SyntaxError):
        # Return an empty list for invalid values
        return []



##########################################################################################################################################

merged_tracks = pd.read_csv('/users/gy22186/mu3e/five_signal_files/predicted_tracksgine_merged_100-0_harsh.csv')

#fake_rates = EvaluateTracks(merged_data, output_dir, truth_condition = 'current_alg')[1]

# Convert nested values to numbers
for col in ['traj_p', 'traj_pt', 'traj_lambda', 'traj_phi']:
    merged_tracks[col] = merged_tracks[col].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x)

print(merged_tracks.columns)
# Preprocess the 'hit_IDs' column
print("Preprocessing 'hit_IDs' column...")
merged_tracks["hit_IDs"] = merged_tracks["hit_IDs"].apply(safe_parse_hit_ids)

# Debugging the preprocessed column
print("Inspecting 'hit_IDs' column after preprocessing:")
print(merged_tracks["hit_IDs"].head(10))
print(merged_tracks["hit_IDs"].apply(type).value_counts())

# Convert hit_IDs back to a list
# merged_tracks["hit_IDs"] = merged_tracks["hit_IDs"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

print(merged_tracks.dtypes)

EfficiencyLambdaMomentumPlot(merged_tracks, min_lam = -1.6,max_lam = 1.6, lam_res = 0.05 , 
                                            min_p = 0, max_p = 60, p_res = 1, p_type = 'traj_pt', num_hits_x = 4)

EfficiencyLambdaMomentumPlot(merged_tracks, min_lam = -1.6,max_lam = 1.6, lam_res = 0.05 , 
                                            min_p = 0, max_p = 60, p_res = 1, p_type = 'traj_p', num_hits_x = 4)

EfficiencyTrackLengthPlot(merged_tracks)

# Calculate total efficiency
total_correct_tracks = merged_tracks["correct_track"].sum()
total_tracks = len(merged_tracks)
total_efficiency = total_correct_tracks / total_tracks if total_tracks > 0 else 0
print(f"\n GNN Total Efficiency: {total_efficiency:.4f} ({total_correct_tracks}/{total_tracks})")

# Load fake rate file and calculate fake rate
fake_rate_file = '/users/gy22186/mu3e/five_signal_files/fake_rate_100-0_harshgine.csv'
fake_rate_data = pd.read_csv(fake_rate_file)

# Calculate and plot fake rate by track length
fake_rate = calculate_fake_rate(fake_rate_file)
FakeRateTrackLengthPlot(fake_rate_data)

