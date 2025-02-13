import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


def TotalEfficiencyMeasure(data_file, truth_measure):
    """Calculates the total efficiency for a data file, accounting for all momenta, angles, and track lengths. Can be done based on 3 variables of truth 
    Defined as: Total no. reconstructed tracks / total no. of reconstructable particles (4+ hits)
    Input:
    - Data file containing:
    -- No. unique particle IDs registering 4 hits (1 per layer... supposedly)
    -- Truth information for unique particle IDs (mommentum)
    -- No. of reconstructed tracks.
    - Measure of what condition on truth is wanted (mc:1, mc_prime:1, 50% rule etc) #TO DO!!
    Output:
    - Overall efficiency measure
    """
    data_file = data_file.copy()

    #Filter the data based on the truth condition used
    if truth_measure == 'absolute':                                             
        data_file['true_tracks'] = ((data_file['mc_prime'] == 1) & (data_file['mc measure'] == 1)).astype(int)


    #Find the efficiency 
    no_reconstructed_tracks = (data_file['true_tracks'] ==1).sum()
    #no_not_reconstructed_tracks = (data_file['true_tracks']==0).sum()
    total_tracks = data_file['tid'].count()
    #tot_track_check = no_reconstructed_tracks + no_not_reconstructed_tracks

    print("Number of tracks correctly reconstructed:", no_reconstructed_tracks)
    #print("Number of tracks not reconstructed:", no_not_reconstructed_tracks)
    print("Total number of reconstructable tracks (non-null rows in the tid column):", total_tracks)
    #print("Check total no. tracks (truth = 0 + truth = 1):", tot_track_check)

    overall_efficiency = (no_reconstructed_tracks)/(total_tracks)
    print("Overall efficiency:",overall_efficiency)
    return 


def efficiencyMomentumPlot(data_file, min_momentum, max_momentum, momentum_res, p_type):
    """Function that produces a plot of track finding algorithm efficiency as a function of total momentum of the underlying truth particle
    Input:
    Data file containing:
    - No. unique particle IDs registering 4 hits (1 per layer... supposedly)
    - Truth information for unique particle IDs (mommentum)
    - No. of reconstructed tracks. 
    Output:
    - Plot of efficiency (normalised) vs truth momentum of the particle IDs
    """
    print("Data loaded successfully for efficiencyMomentumPlot")

    data = data_file.copy()


#### Edit this bit 
    momentum_bin_edges = np.arange(min_momentum, (max_momentum + momentum_res), momentum_res)
    bin_labels = (momentum_bin_edges[:-1] + momentum_bin_edges[1:]) / 2  # Compute bin centers
    # Bin the 'traj_p' values into categories
    data.loc[:,'momentum_bin'] = pd.cut(data_file[p_type], bins=momentum_bin_edges, labels = bin_labels, include_lowest=True)

    # Compute efficiency per bin
    efficiency_data = data.groupby('momentum_bin').agg(
        total_tracks=('tid', 'count'),
        reconstructed_tracks=('mc_tid', lambda x: x.notna().sum())
    )
    
    efficiency_data['efficiency'] = efficiency_data['reconstructed_tracks'] / efficiency_data['total_tracks']
    
    # Plot efficiency vs momentum
    plt.figure(figsize=(8, 6))
    plt.plot(efficiency_data.index, efficiency_data['efficiency'], marker='o', linestyle='-', color='b', label='Efficiency')
    plt.xlabel('Momentum (traj_p)')
    plt.ylabel('Efficiency')
    plt.title('Track Reconstruction Efficiency vs Momentum')
    plt.ylim(0, 1.05)  # Keep efficiency between 0 and 1
    plt.xticks(bin_labels)  # Use bin centers for x-axis labels
    plt.grid()
    plt.legend()
    plt.show()

    return


def efficiencyTrackLengthPlot(data_file, truth_measure):
    """Function to plot the efficiency of the track finding as a function of the number of registered hits. Can be done for variable truth definitions
    Input:
    - Data file containing:
        -- No. unique particle IDs 
        -- Truth information for unique particle IDs (mommentum)
        -- No. of reconstructed tracks.
    - Measure of what condition on truth is wanted (mc:1, mc_prime:1, 50% rule etc) #TO DO!!
        Output:
    - Overall efficiency measure
    """ 



def efficiency_lambda_p_plot(data_file):
    """Function to produce a 2D efficiency heatmap (traj_lambda vs traj_p) using Matplotlib"""

    df = data_file.copy()  # Work on a copy to avoid modifying the original dataframe

    # Define bin edges for lambda and momentum (adjust based on your data range)
    lambda_bins = np.linspace(-1.5, 1.5, 50)  # 20 bins for lambda
    momentum_bins = np.linspace(0, 100, 50)  # 20 bins for momentum from 0 to 100

    # Assign bins
    df['lambda_bin'] = pd.cut(df['traj_lambda'], bins=lambda_bins, labels=False)
    df['momentum_bin'] = pd.cut(df['traj_p'], bins=momentum_bins, labels=False)

    # Count total occurrences in each bin
    total_counts = df.groupby(['lambda_bin', 'momentum_bin'])['tid'].count().reset_index(name='total_tracks')

    # Count reconstructed tracks (non-null mc_tid)
    reconstructed_counts = df.dropna(subset=['mc_tid']).groupby(['lambda_bin', 'momentum_bin'])['tid'].count().reset_index(name='reconstructed_tracks')

    # Merge both dataframes to align bins
    efficiency_data = total_counts.merge(reconstructed_counts, on=['lambda_bin', 'momentum_bin'], how='left').fillna(0)

    # Compute efficiency
    efficiency_data['efficiency'] = efficiency_data['reconstructed_tracks'] / efficiency_data['total_tracks']

    # Pivot efficiency data into a 2D grid
    lambda_indices = efficiency_data['lambda_bin'].astype(int).values
    momentum_indices = efficiency_data['momentum_bin'].astype(int).values
    efficiency_values = efficiency_data['efficiency'].values

    # Create a 2D array for efficiency
    efficiency_grid = np.full((len(momentum_bins)-1, len(lambda_bins)-1), np.nan)  # Fill with NaNs initially
    efficiency_grid[momentum_indices, lambda_indices] = efficiency_values  # Fill known efficiency values

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(lambda_bins, momentum_bins, efficiency_grid.T, shading='auto', cmap='viridis', vmin=0, vmax=1)

    # Color bar
    cbar = plt.colorbar()
    cbar.set_label("Efficiency")

    # Labels
    plt.xlabel("Lambda (λ)")
    plt.ylabel("Momentum (p)")
    plt.title("Efficiency Heatmap (λ vs p)")

    plt.show()




sort_file = "/root/Mu3eProject/WorkingVersion/Mu3eProject/DataFilesV5.3/signal1_99_32652/hits_data_signal1_99_32652_with_mcinfo_tid_sorted.csv"
sort_mc_file = "/root/Mu3eProject/WorkingVersion/Mu3eProject/DataFilesV5.3/signal1_99_32652/mc_truth_signal1_99_32652.csv"
trirec_file = "/root/Mu3eProject/RawData/TrirecFiles/signal1_99_32652/trirec_data_signal1_99_32652_frames.csv"


#Investigating the 3 different files of tids we have
hits_data = pd.read_csv(sort_file)
hits_traj_mc_data = pd.read_csv(sort_mc_file)
trirec_data = pd.read_csv(trirec_file)

#Finding Number of unique tid entries in each
n_tid_hits = len(pd.unique(hits_data['tid']))
n_tid_hits_traj_mc = len(pd.unique(hits_traj_mc_data['traj_ID']))
n_tid_trirec = len(pd.unique(trirec_data['mc_tid']))

#####################################################################################
# # Filter the hits TID counts down to those that appear with more than 4 hits
# tid_counts = hits_data['tid'].value_counts()
# valid_tids = tid_counts[tid_counts >= 4].index

# # Get the total number of unique TIDs 
# valid_tids_len= len(valid_tids)

# # Filter hits_data to keep only rows with those TIDs matching the >4 criteria
# filtered_hits_data = hits_data[hits_data['tid'].isin(valid_tids)]
# unique_tids_filtered = filtered_hits_data[['tid']].drop_duplicates() # Extract unique TIDs from the filtered data, drop the duplicates
########################################################################################


# Count occurrences of each 'tid' and store it in a new column
hits_data['no_hits'] = hits_data.groupby('tid')['tid'].transform('count')

# Remove duplicates while keeping the 'no_hits' count
unique_tids = hits_data[['tid', 'no_hits']].drop_duplicates()

# Apply condition to filter based on 'no_hits' (e.g., keep only TIDs with 4+ hits)
filtered_hits_data = unique_tids[unique_tids['no_hits'] >= 4]





# Compute the total momentum for the traj_mc data (Needed for hits tid where no trirec reconstruction)
hits_traj_mc_data['traj_p'] = np.sqrt(
    hits_traj_mc_data['traj_px']**2 + 
    hits_traj_mc_data['traj_py']**2 + 
    hits_traj_mc_data['traj_pz']**2
)

#Find pt
hits_traj_mc_data['traj_pt'] = np.sqrt(
    hits_traj_mc_data['traj_px']**2 + 
    hits_traj_mc_data['traj_py']**2
)

#Find Lambda angle
hits_traj_mc_data['traj_lambda'] = np.arctan2(hits_traj_mc_data['traj_pz'], hits_traj_mc_data['traj_pt'])




# Merge with hits_traj_mc_data to get momentum information

hits_traj_mc_data_unique = hits_traj_mc_data.drop_duplicates(subset=['traj_ID'])

merged_hits_traj_mc = filtered_hits_data.merge(
    hits_traj_mc_data_unique[['traj_ID', 'traj_px', 'traj_py', 'traj_pz', 'traj_vx', 'traj_vy', 'traj_vz', 'traj_p', 'traj_pt', 'traj_lambda']],
    left_on='tid',
    right_on='traj_ID',
    how='left'
)

# Drop duplicate 'traj_ID' column since it's just 'tid' renamed
merged_hits_traj_mc.drop(columns=['traj_ID'], inplace=True)


# Merge with trirec data using a left join to keep all tid's in merged_df
trirec_data_unique = trirec_data.drop_duplicates(subset=['mc_tid'])

all_merged = merged_hits_traj_mc.merge(
    trirec_data_unique[['mc_prime', 'mc measure', 'mc_tid', 'mc_vx', 'mc_vy', 'mc_vz', 'mc_p','mc_pt', 'mc_lam']],  # Keep only relevant columns
    left_on='tid',
    right_on='mc_tid',
    how='left'  # Keeps all rows from merged_df, fills missing trirec info with NaN
)

# Drop all the rows where no traj_mc info present (i.e where a hit tid but does not corr to track NOR truth particle)
df_merged_truths = all_merged.dropna(subset = ['traj_px'])

# Print results
print("Final dataframe with merged info:")
print(df_merged_truths.head())  # Print first few rows to check








#Save the final data as a csv for investigating
file_path = "/root/Mu3eProject/RawData/tid_data_signal1_99_drop_notruths.csv"
df_merged_truths.to_csv(file_path, index=False)




#efficiencyMomentumPlot(df_merged_truths, min_momentum = 0, max_momentum = 100, momentum_res = 1, p_type = 'traj_p')
TotalEfficiencyMeasure(df_merged_truths, truth_measure = 'absolute')

efficiency_lambda_p_plot(df_merged_truths)

# Verifying prints on TID info:
# print("No. of unique TID in hits data:",n_tid_hits)
# print("Number of unique TIDs with 4+ occurrences:", valid_tids_len)
# print("No. of unique traj_ID in mu3e_mc_truth data:", n_tid_hits_traj_mc)
# print("No. of unique TID in trirec data:",n_tid_trirec)