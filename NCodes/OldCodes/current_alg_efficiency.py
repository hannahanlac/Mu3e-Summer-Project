import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors


def BuildComparisonData(hits_file, hits_mc_file, trirec_file, signal_no):    
    """Function that takes the relevant data files for hits, their ground truth values, and assigned tracks,   
    and builds a single data file for evaluation.
    Input:
    - hits_file: File of raw hits data and their tids (the true track they belong to)
    - hits_mc_file: File of the tid track truth info (from the _sort file mc info: denoted traj_... )
    - trirec_file: File containing trirec reconstruction data from current algorithm, and its truth parameters. 
    #NOTE: If function changed to work for output of the ML model, will need to change trirec_file to be OUTPUT FILE OF ML MODEL.
    Output:
    - Single file containing Info on no. of unique 4-hit reconstructable tracks, ground truth momenta of tracks, and reconstruction attempt
    """

    #Open the 3 different files and read the csvs
    hits_data = pd.read_csv(hits_file)
    hits_traj_mc_data = pd.read_csv(hits_mc_file)
    trirec_data = pd.read_csv(trirec_file)

    # Compute traj_mc track info: total p, pt, and lambda angle.(Needed for hits tid where no trirec reconstruction)
    hits_traj_mc_data['traj_p'] = np.sqrt(
        hits_traj_mc_data['traj_px']**2 + 
        hits_traj_mc_data['traj_py']**2 + 
        hits_traj_mc_data['traj_pz']**2
    )

    hits_traj_mc_data['traj_pt'] = np.sqrt(
        hits_traj_mc_data['traj_px']**2 + 
        hits_traj_mc_data['traj_py']**2
    )

    hits_traj_mc_data['traj_lambda'] = np.arctan2(hits_traj_mc_data['traj_pz'], hits_traj_mc_data['traj_pt'])


    # Count no. of each tid in hit data, remove duplicates, filter out all with <4 hits (our condition 'reconstructable')
    hits_data['num_hits'] = hits_data.groupby('tid')['tid'].transform('count')
    unique_tids = hits_data[['tid', 'num_hits']].drop_duplicates()
    filtered_hits_data = unique_tids[unique_tids['num_hits'] >= 4]


    #Count no. of reconstructed tracks for single tid in trirec file. Then remove duplicates (SEE NOTE)
    trirec_data['num_recon_tracks'] = trirec_data.groupby('mc_tid')['mc_tid'].transform('count')
  #trirec_data.drop_duplicates(subset=['mc_tid']) 
    # NOTE:This line removes duplicates of mc_tid. You want to REMOVE duplicates for the overall efficiency, but KEEP duplicates for the fake rate! 
    # For now: I simply keep the duplicates here, then remove them in the function for overall efficiency


    # Merge hits_data with hits_traj_mc_data to get momentum information for tids
    hits_traj_mc_data_unique = hits_traj_mc_data.drop_duplicates(subset=['traj_ID'])

    merged_hits_traj_mc = filtered_hits_data.merge(
        hits_traj_mc_data_unique[['traj_ID', 'traj_px', 'traj_py', 'traj_pz', 'traj_vx', 'traj_vy', 'traj_vz', 'traj_p', 'traj_pt', 'traj_lambda']],
        left_on='tid',
        right_on='traj_ID',
        how='left'
    )
    # Drop all the rows where no traj_mc info present (i.e where a hit tid but does not corr to track NOR truth particle), then drop traj_ID as is tid
    merged_hits_traj_mc.drop(columns=['traj_ID'], inplace=True)

    # Merge with trirec data 
    all_merged = merged_hits_traj_mc.merge(
        trirec_data[['length', 'mc_prime', 'mc measure', 'mc_tid', 'num_recon_tracks', 'mc_vx', 'mc_vy', 'mc_vz', 'mc_p','mc_pt', 'mc_lam']],  # Keep only relevant columns
        left_on='tid',
        right_on='mc_tid',
        how='left'  # Keeps all rows from merged_df, fills missing trirec info with NaN
    )

    df_merged_truths = all_merged.dropna(subset = ['traj_px'])

    # # Print results
    # print("Final dataframe with merged info:")
    # print(df_merged_truths.head())  # Print first few rows to check

    file_path = f"/root/Mu3eProject/RawData/ComparisonData/comparison_data_{signal_no}.csv"
    df_merged_truths.to_csv(file_path, index=False)
    return 

#Efficiency measures
def TotalEfficiencyMeasure(data_file, truth_measure, hit_count):
    """Calculates the total efficiency for a data file, accounting for all momenta, angles, and track lengths. Can be done based on 3 variables of truth 
    Defined as: Total no. reconstructed tracks / total no. of reconstructable particles (4+ hits)
    Input:
    - Data file containing:
    -- No. unique particle IDs registering 4 hits (1 per layer... supposedly)
    -- Truth information for unique particle IDs (mommentum)
    -- No. of reconstructed tracks.
    - Measure of what condition on truth is wanted (mc:1, mc_prime:1, 50% rule etc) #TO DO!!
    - hit_count: Number of registered hits count
    Output:
    - Overall efficiency measure
    """
    data_file = data_file.copy()
    data_file = data_file[data_file['num_hits'] >= hit_count] 

    if truth_measure == 'absolute':  
        data_file['abs_true_track'] = ((data_file['mc_prime'] == 1) & (data_file['mc measure'] == 1)).astype(int)
        data_file = data_file.sort_values(by='abs_true_track', ascending=False)

        num_true_inc_duplicates = (data_file['abs_true_track'] ==1).sum() #NOTE: Includes duplicates where 2+ absolute true for same tid
        data_file = data_file.drop_duplicates(subset='tid', keep="first") 
        num_true_no_duplicates = (data_file['abs_true_track'] ==1).sum() #Does not include duplicates: only 1 track entry per tid
        total_tracks = data_file['tid'].nunique()

        print("Number of tracks correctly reconstructed, inc duplicates:", num_true_inc_duplicates)
        print("no. tracks ex duplicates:", num_true_no_duplicates)
        print("Total number of reconstructable tracks:", total_tracks)

        overall_efficiency_inc_dups = (num_true_inc_duplicates)/(total_tracks)
        overall_efficiency_no_dups = (num_true_no_duplicates)/(total_tracks)
        print("Overall efficiency including duplicates:",overall_efficiency_inc_dups)
        print("Overall efficiency excluding duplicates:", overall_efficiency_no_dups)
    return 

def EfficiencyTrackLengthPlot(data_file, truth_measure,):
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

    data_file = data_file.copy()
    data_file['abs_true_track'] = ((data_file['mc_prime'] == 1) & (data_file['mc measure'] == 1)).astype(int)
    data_file = data_file.sort_values(by='abs_true_track', ascending=False)
    data_file = data_file.drop_duplicates(subset='tid', keep="first") 
    data_file = data_file.sort_values(by='num_hits', ascending = True) 
    #NOTE: This cuts all the duplicate tid entries, removing also any duplicate tracks created which both have a 'abs_true_track' value of 1. 
    #(See labbook 13/02/2025 daynotes for reasoning)

 
    bins = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, float('inf')]  # 10+ is grouped as the last bin
    bin_labels = ['4', '5', '6', '7', '8', '9','10', '11', '12', '12+']
    data_file['hit_bin'] = pd.cut(data_file['num_hits'], bins=bins, labels=bin_labels, right=True)

    # elif method =='length': NOTE: Messing around if need to plot as 'length of reconstructed track' efficiency. Think redundant
    #     grouped_data_tracks = data_file.groupby('length,', observed = False)
    #     total_tracks_per_length = grouped_data_tracks['mc_tid'].count()
    #     reconstructed_tracks_abs_true = grouped_data_tracks['abs_true_track'].sum()


    # Group data by 'hit_bin' and calculate efficiency for each bin
    grouped_data_tracks = data_file.groupby('hit_bin', observed=False)
    total_tracks_per_length = grouped_data_tracks['tid'].count()
    reconstructed_tracks_per_hit = grouped_data_tracks['mc_tid'].count() # Total tracks per group - Only counts where a reconstruction was made.
    reconstructed_tracks_abs_true = grouped_data_tracks['abs_true_track'].sum() #Counts the total complete true tracks

    efficiency_all = reconstructed_tracks_per_hit / total_tracks_per_length #Efficiency for all recon tracks
    efficiency_abs_true = reconstructed_tracks_abs_true / total_tracks_per_length #Efficiency for only abs_true tracks

    if truth_measure == 'absolute':
        plt.figure(figsize=(10, 6))
        plt.bar(efficiency_abs_true.index, efficiency_abs_true.values, color='blue', edgecolor = 'black', label='Efficiency')
        plt.xlabel("Track Length", fontsize=14)
        plt.ylabel("Efficiency", fontsize=14)
        plt.title("Efficiency as a Function of Track Length", fontsize=18)
        plt.grid(True)
        plt.legend()
        plt.yticks(np.arange(0, 1.1, 0.1))  
        plt.show()

    else:
        plt.figure(figsize=(10, 6))
        plt.bar(efficiency_all.index, efficiency_all.values, color='g', edgecolor= 'black' ,label='Efficiency')
        plt.xlabel("Track Length (num_hits) Bins")
        plt.ylabel("Efficiency")
        plt.title("Efficiency as a Function of Track Length")
        plt.grid(True)
        plt.legend()
        plt.yticks(np.arange(0, 1.1, 0.1))  
        plt.show()   

    return

def EfficiencyMomentumPlot(data_file, min_momentum, max_momentum, momentum_res, p_type):#Needs editing
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
    efficiency_data = data.groupby('momentum_bin', observed = False).agg(total_tracks=('tid', 'count'),reconstructed_tracks=('mc_tid', lambda x: x.notna().sum()),
    )
    
    efficiency_data['efficiency'] = efficiency_data['reconstructed_tracks'] / efficiency_data['total_tracks']
    
    # Plot efficiency vs momentum
    plt.figure(figsize=(8, 6))
    plt.plot(efficiency_data.index, efficiency_data['efficiency'], marker='o', linestyle='-', color='b', label='Efficiency')
    plt.xlabel('Momentum (traj_p)')
    plt.ylabel('Efficiency')
    plt.title('Track Reconstruction Efficiency vs Momentum')
    plt.ylim(0, 1)  # Keep efficiency between 0 and 1
    plt.xticks(bin_labels)  # Use bin centers for x-axis labels
    plt.grid()
    plt.legend()
    plt.show()

    return

def EfficiencyLambdaMomentumPlot(data_file, min_lam, max_lam, lam_res, min_p, max_p, p_res, p_type, num_hits, truth_measure):
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

    data_file = data_file.copy()
    data_file['abs_true_track'] = ((data_file['mc_prime'] == 1) & (data_file['mc measure'] == 1)).astype(int)
    data_file = data_file.sort_values(by='abs_true_track', ascending=False)
    data_file = data_file.drop_duplicates(subset='tid', keep="first") 
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

    # Group data and ENSURE ALL BINS INCLUDED
    if truth_measure == 'absolute':
        grouped = data_file.groupby([ 'lambda_bin','momentum_bin']).agg(
            total_tracks=('tid', 'count'),
            true_tracks=('abs_true_track', 'sum')
        ).reindex(bin_index, fill_value=0)  # Ensure no bins dropped

    else:
        grouped = data_file.groupby([ 'lambda_bin','momentum_bin']).agg(
            total_tracks=('tid', 'count'),
            true_tracks=('mc_tid', 'count')
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

#Fake rate measures
def TotalFakeRateMeasure(data_file, fake_measure, num_hits):
    """Calculates the total fake rate for a data file, accounting for all momenta, angles, and track lengths. Can be done based on 2 definitions of fake rate 
    'harsh': Any track where not 1 'abs_true_track'
    'lenient': Any track where both mc_prime and mc measure are 0.
    Input:
    - Data file containing:
    -- No. unique particle IDs registering 4 hits (1 per layer... supposedly)
    -- Truth information for unique particle IDs (mommentum)
    -- Total No. of reconstructed tracks, including all fakes and duplicates
    - Measure of what condition of fake_rate is wanted (harsh, lenient)
    - hit_count: Number of registered hits count
    Output:
    - Overall fake rate percentage for track
    """
    assert fake_measure in ['harsh', 'lenient'], "fake_measure must be 'harsh' or 'lenient'"
    data_file = data_file.copy()
    data_file['abs_true_track'] = ((data_file['mc_prime'] == 1) & (data_file['mc measure'] == 1)).astype(int)
    data_file = data_file[data_file['num_hits'] >= num_hits]  # Allows control here of if want to do for just longer tracks. Leave at 4 for no change
    data_file = data_file.dropna(subset=['mc_tid']) # Drop all non-reconstructed particles, keep only 'reconstructed' tracks

    if fake_measure == 'harsh':
        total_tracks = len(data_file) # Use length as there are duplicates of tid for duplicate tracks, want to include them all
        num_fake_tracks = (data_file['abs_true_track'] ==0).sum()
        print(f"total tracks: {total_tracks}")
        print(f"number of fake tracks:{num_fake_tracks}")

    elif fake_measure == 'lenient':
        total_tracks = len(data_file) #see above
        num_fake_tracks = ((data_file['mc_prime'] == 0) & (data_file['mc measure'] == 0)).sum()
        print(f"total tracks: {total_tracks}")
        print(f"number of fake tracks:{num_fake_tracks}")

    total_fake_rate = num_fake_tracks/total_tracks
    print(f"Overall {fake_measure} fake rate: {total_fake_rate}")

    return

def FakeRateTrackLengthPlot(data_file, fake_measure, num_hits):
    """Function to plot the fake rate of the track finding as a function of the reconstructed track length. Can be done for different fake rate definitions.
    'harsh': Any track where not 1 'abs_true_track'
    'lenient': Any track where both mc_prime and mc measure are 0.
    Input:
    - Data file containing:
        -- No. unique particle IDs 
        -- Truth information for unique particle IDs
        -- No. of reconstructed tracks.
        -- Length of track: 4, 6, 8 (derived from Trirec file)
    - Measure of what condition on truth is wanted (mc:1, mc_prime:1, 50% rule etc) #TO DO!!
    Output:
    - Overall efficiency measure
    """
    assert fake_measure in ['harsh', 'lenient'], "fake_measure must be 'harsh' or 'lenient'"
    data_file = data_file.copy()
    data_file['abs_true_track'] = ((data_file['mc_prime'] == 1) & (data_file['mc measure'] == 1)).astype(int)
    data_file = data_file[data_file['num_hits'] >= num_hits]  # Allows control here of if want to do for just longer tracks. Leave at 4 for no change
    data_file = data_file.dropna(subset=['mc_tid']) # Specific for this function, as only considering reconstructed tracks (remove non-recon ones)

    grouped_data_tracklength = data_file.groupby('length', observed=False)
    total_tracks_per_length = grouped_data_tracklength['tid'].count() #Again count total, including the duplicates

    if fake_measure == 'harsh':
        fake_tracks = grouped_data_tracklength['abs_true_track'].agg(lambda x: (x == 0).sum())

    elif fake_measure == 'lenient':
        fake_tracks = grouped_data_tracklength.apply(lambda x: ((x['mc_prime'] == 0) & (x['mc measure'] == 0)).sum())
    
    fake_rate = fake_tracks / total_tracks_per_length

    # Print fake rates per track length
    print("\nFake Rate per Track Length:")
    for length, rate in fake_rate.items():
        print(f"Track Length {length}: Fake Rate = {rate:.4f}")

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(fake_rate.index, fake_rate.values, color='red', edgecolor = 'black', alpha=0.8)

    # Add text labels above bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.3f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Labels and title
    plt.xlabel("Track Length", fontsize=14)
    plt.ylabel("Fake Rate", fontsize=14)
    plt.title(f"Fake Rate vs Track Length ({fake_measure.capitalize()})", fontsize=16)
    plt.xticks(fake_rate.index)  # Ensure track lengths are used as x-ticks
    plt.ylim(0, max(fake_rate.values) * 1.2)  # Add some space above the highest bar

    plt.show()

    return fake_rate  # Returning it in case you want to use it later

def FakeRateLambdaMomentumPlot(data_file, min_lam, max_lam, lam_res, min_p, max_p, p_res, p_type, num_hits, fake_measure):
    """Function that produces a plot of the fake rate (no. fake tracks/total reconstructed) in the data as a function of total momentum
       and inclination angle (lambda) of the truth particles. Creates a 2D colourmap of fake rate as output.
    Input:
    - Data file containing:
        - No. unique particle IDs registering 4 hits (1 per layer... supposedly)
        - Truth information for unique particle IDs (mommentum and lambda)
        - No. of reconstructed tracks. 
        - Mc truth measure of the track ('abs' vs not.)
    - Lambda and Momentum ranges, and resolutions
    - p_type: traj_p (total p) or traj_pt (transverse)
    - num_hits: Min num hits used (4 by default for all 'reconstructable' tracks)
    Output:
    - Colourmap of fake rate as function of truth momentum and lambda of the particle IDs
    """
    assert fake_measure in ['harsh', 'lenient'], "fake_measure must be 'harsh' or 'lenient'"

    data_file = data_file.copy()
    data_file['abs_true_track'] = ((data_file['mc_prime'] == 1) & (data_file['mc measure'] == 1)).astype(int)
    data_file = data_file[data_file['num_hits'] >= num_hits]  # Allows control here of if want to do for just longer tracks. Leave at 4 for no change
    data_file = data_file.dropna(subset=['mc_tid']) # Drop all non-reconstructed particles, keep only 'reconstructed' tracks

    #data_file = data_file.drop_duplicates(subset='tid', keep="first") NOT THIS! Remove: We want to keep the duplicates for analysing
    #data_file = data_file.sort_values(by='abs_true_track', ascending=False) No need for this as above

    file_path = "/root/Mu3eProject/RawData/ComparisonData/Fake_rate_test_1.csv"
    data_file.to_csv(file_path, index=False)

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

    if fake_measure == 'harsh':
        grouped = data_file.groupby([ 'lambda_bin','momentum_bin']).agg(
            total_tracks=('tid', 'count'), # Just use length of file as we want 
            fake_tracks=('abs_true_track', lambda x: (x == 0).sum())
        ).reindex(bin_index, fill_value=0)  # Ensure no bins dropped

    elif fake_measure == 'lenient':
        grouped = data_file.groupby([ 'lambda_bin','momentum_bin']).agg(
            total_tracks=('tid', 'count'), # Just use length of file as we want 
            fake_tracks=('mc_prime', lambda x: ((x == 0) & (data_file.loc[x.index, 'mc measure'] == 0)).sum())
        ).reindex(bin_index, fill_value=0)  # Ensure no bins dropped


    # Efficiency and efficiency matrix of correct dimensions
    grouped['fake_rate'] = grouped['fake_tracks'] / grouped['total_tracks'].replace(0, np.nan) 
    fake_rate_matrix = grouped['fake_rate'].unstack().fillna(0)

    if p_type == 'traj_pt':
        p_plot_name = 'pt'
    elif p_type == 'traj_p':
        p_plot_name = 'p'
    
    #Plot
    plt.figure(figsize=(10, 6))

    levels = np.arange(0, 1.05, 0.05) # Use for controlling colourmap no. colours
    level_ticks = np.arange(0,1.1, 0.1)
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=256)
    mesh = plt.pcolormesh(lambda_bins, momentum_bins, fake_rate_matrix.T, cmap='viridis', norm=norm, shading='auto')
    cbar = plt.colorbar(mesh, ticks=level_ticks, pad=0.0)
    cbar.set_label('Fake Rate', fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=16, length=8, width=2) 
    ax.tick_params(axis='both', which='minor', labelsize=12, length=4, width=1)  
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    plt.xlabel("$\lambda$ [rad]", fontsize=18)
    plt.ylabel(f"{p_plot_name} [MeV/c]", fontsize=18)
    plt.title(f"Fake Rate as a Function of $\lambda$ and {p_plot_name}", fontsize=20)

    plt.show()
    return

#Other measures
def TrackFrequencyLambdaMomentumPlot(data_file, min_lam, max_lam, lam_res, min_p, max_p, p_res, p_type, num_hits, truth_measure):
    """Function that produces a plot of the frequency of hits in the data as a function of total momentum
       and inclination angle (lambda) of the truth particles. Creates a 2D heatmap of intensity as output.
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
    - Heatmap of frequency as function of truth momentum and lambda of the particle IDs
    """

    data_file = data_file.copy()
    data_file['abs_true_track'] = ((data_file['mc_prime'] == 1) & (data_file['mc measure'] == 1)).astype(int)
    data_file = data_file.sort_values(by='abs_true_track', ascending=False)
    data_file = data_file.drop_duplicates(subset='tid', keep="first") 
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
        total_tracks=('tid', 'nunique'),
    ).reindex(bin_index, fill_value=0)  # Ensure no bins dropped

    # Efficiency and efficiency matrix of correct dimensions
    grouped['Frequency'] = grouped['total_tracks'] 
    frequency_matrix = grouped['Frequency'].unstack().fillna(0)

    if p_type == 'traj_pt':
        p_plot_name = 'pt'
    elif p_type == 'traj_p':
        p_plot_name = 'p'
    
    #Plot
    plt.figure(figsize=(10, 6))

    levels = np.linspace(0, grouped['Frequency'].max(), 11)
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=256)
    mesh = plt.pcolormesh(lambda_bins, momentum_bins, frequency_matrix.T, cmap='inferno', norm=norm, shading='auto')
    cbar = plt.colorbar(mesh, ticks=levels, pad=0.0)
    cbar.set_label('Frequency', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14, length=8, width=2) 
    ax.tick_params(axis='both', which='minor', labelsize=10, length=4, width=1)  
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    plt.xlabel("$\lambda$ [rad]", fontsize=16)
    plt.ylabel(f"{p_plot_name} [MeV/c]", fontsize=16)
    plt.title(f"Frequency as a Function of $\lambda$ and {p_plot_name}", fontsize=18)

    plt.show()
    return

def RatioLongToShortTracks (data_file, min_lam, max_lam, lam_res, min_p, max_p, p_res, p_type, num_hits):
    """Function that produces a plot of the ratio of long : short tracks reconstructed in the data as a function of total momentum
       and inclination angle (lambda) of the truth particles. Creates a 2D colourmap of ratio long:short as output.
    Input:
    - Data file containing:
        - No. unique particle IDs registering 4 hits (1 per layer... supposedly)
        - Truth information for unique particle IDs (mommentum and lambda)
        - No. of reconstructed tracks. 
        - The length of the reconstructed track (from trirec file)
    - Lambda and Momentum ranges, and resolutions
    - p_type: traj_p (total p) or traj_pt (transverse)
    - num_hits: Min num hits used (4 by default for all 'reconstructable' tracks)
    - truth_measure: 'absolute' for strongest, otherwise all reconstructed tracks considered
    Output:
    - Colourmap of ratio long:short as function of truth momentum and lambda of the particle IDs
    """

    data_file = data_file.copy()
    data_file['abs_true_track'] = ((data_file['mc_prime'] == 1) & (data_file['mc measure'] == 1)).astype(int)
    data_file = data_file.sort_values(by='abs_true_track', ascending=False)
    data_file = data_file.drop_duplicates(subset='tid', keep="first") 
    data_file = data_file[data_file['num_hits'] >= num_hits]  # Allows control here of if want to do for just longer tracks. Leave at 4 for no change
    data_file = data_file.dropna(subset=['mc_tid']) # Specific for this function, as only considering reconstructed tracks (remove non-recon ones)


    print("no.unique tids:", data_file['tid'].nunique())
    print("no.entries:",len(data_file))

    # Define bins for lambda and momentum
    lambda_bins = np.arange(min_lam, max_lam + lam_res, lam_res)
    momentum_bins = np.arange(min_p, max_p + p_res, p_res)

    # Assign bin indices
    data_file['lambda_bin'] = np.digitize(data_file['traj_lambda'], lambda_bins) - 1 #-1 to ensure from 0 indexing
    data_file['momentum_bin'] = np.digitize(data_file[p_type], momentum_bins) - 1
    data_file['lambda_bin'] = np.clip(data_file['lambda_bin'], 0, len(lambda_bins) - 2)
    data_file['momentum_bin'] = np.clip(data_file['momentum_bin'], 0, len(momentum_bins) - 2)

    # All possible bin combinations indexed. NOTE: This is needed to ensure no bins are dropped later! (This was found out the hard way...)
    bin_index = pd.MultiIndex.from_product(
        [range(len(lambda_bins) - 1), range(len(momentum_bins) - 1)],
        names=[ 'lambda_bin','momentum_bin']
    )

    # Group data and ENSURE ALL BINS INCLUDED
    grouped = data_file.groupby(['lambda_bin', 'momentum_bin']).agg(
        short_tracks=('length', lambda x: (x == 4).sum()),   # Count num_hits == 4
        long_tracks=('length', lambda x: ((x == 6) | (x == 8)).sum())  # Count num_hits == 6 or 8
    ).reindex(bin_index, fill_value=0)  # Ensure all bins are included


    # Compute the ratio (long / short), avoiding division by zero
    grouped['ratio'] = grouped['long_tracks'] / grouped['short_tracks'].replace(0, np.nan)

    # Convert to matrix format for plotting
    ratio_matrix = grouped['ratio'].unstack(fill_value=0)

    if p_type == 'traj_pt':
        p_plot_name = 'pt'
    elif p_type == 'traj_p':
        p_plot_name = 'p'

    plt.figure(figsize=(10, 6))

    levels = np.arange(0, 1.05, 0.05) # Use for controlling colourmap no. colours
    level_ticks = np.arange(0,1.1, 0.1)
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=256)
    mesh = plt.pcolormesh(lambda_bins, momentum_bins, ratio_matrix.T, cmap='viridis', norm=norm, shading='auto')
    cbar = plt.colorbar(mesh, ticks=level_ticks, pad=0.0)
    cbar.set_label('Ratio long : short', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14, length=8, width=2) 
    ax.tick_params(axis='both', which='minor', labelsize=10, length=4, width=1)  
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    plt.xlabel("$\lambda$ [rad]", fontsize=16)
    plt.ylabel(f"{p_plot_name} [MeV/c]", fontsize=16)
    plt.title(f"Ratio long:short tracks as a Function of $\lambda$ and {p_plot_name}", fontsize=18)

    plt.show()
    return


# Define the files for building your data, build dataset
hits_data = "/root/Mu3eProject/RawData/TransformerData/signal1_95/signal1_95_hits_data.csv"
truth_data = "/root/Mu3eProject/RawData/TransformerData/signal1_95/signal1_95_truth_data.csv"
trirec_file = "/root/Mu3eProject/RawData/TrirecFiles/signal1_95_32652/trirec_data_signal1_95_32652_frames.csv"
#BuildComparisonData(hits_data, truth_data, trirec_file, signal_no = 'signal1_95')


#Test efficiency
comparison_file = "/root/Mu3eProject/RawData/ComparisonData/comparison_data_signal1_98.csv"
df_comparison = pd.read_csv(comparison_file)


# EfficiencyTrackLengthPlot(df_comparison, truth_measure='absolute')
#TotalEfficiencyMeasure(df_comparison, truth_measure = 'absolute', hit_count = 4)
# EfficiencyMomentumPlot(df_comparison, min_momentum = 0, max_momentum = 100, momentum_res = 10, p_type = 'traj_p')

#EfficiencyLambdaMomentumPlot(df_comparison, min_lam = -1.6,max_lam = 1.6, lam_res = 0.05 , min_p = 0, max_p = 60, p_res = 1, p_type = 'traj_pt', num_hits = 4, truth_measure = 'absolute')
EfficiencyLambdaMomentumPlot(df_comparison, min_lam = -1.6,max_lam = 1.6, lam_res = 0.05 , min_p = 0, max_p = 60, p_res = 1, p_type = 'traj_p', num_hits = 4, truth_measure = 'all')

# RatioLongToShortTracks(df_comparison, min_lam = -1.6,max_lam = 1.6, lam_res = 0.05 , min_p = 0, max_p = 60, p_res = 1, p_type = 'traj_pt', num_hits = 4)

# TrackFrequencyLambdaMomentumPlot(df_comparison, min_lam = -1.6,max_lam = 1.6, lam_res = 0.05 , min_p = 0, max_p = 60, p_res = 2, p_type = 'traj_p', num_hits = 4, truth_measure = 'absolute')

FakeRateLambdaMomentumPlot(df_comparison, min_lam = -1.6,max_lam = 1.6, lam_res = 0.05 , min_p = 0, max_p = 60, p_res = 2, p_type = 'traj_p', num_hits = 4, fake_measure = 'harsh') #Harsh = non absolute true = fake, lenient = mc_prime:0 AND mc measure:0

#TotalFakeRateMeasure(df_comparison, fake_measure = 'lenient', num_hits = 4 )

#FakeRateTrackLengthPlot(df_comparison, fake_measure = 'lenient', num_hits = 4)

