import uproot
import pandas as pd
import numpy as np
import awkward as ak
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
import glob
from tqdm import tqdm

###### Use this code to create the data sets for the transformer model. Saves all the files into a data_training folder.

# Hit data extraction
class Hit(object):
    """Decodes a 32 bit hit ID (as found in the 'hit_pixelid' branch) into its constituent parts"""
    def __init__(self, hitIndex):
        self.hitIndex = hitIndex
    def __str__(self):
        return "%d (station %d, layer %d, ladder %d, chip %d, pixel [%d, %d])" % (self.hitIndex, self.station(), self.layer(), self.phi(), self.z(), self.x(), self.y())
    def chipid(self): return self.hitIndex >> 16
    def pixelid(self): return self.hitIndex
    def column(self):
        """The raw pixel column relative to the left side of the chip. Different layers orient chips differently, so "left" is not consistent."""
        return (self.hitIndex >> 8) & 0xFF
    def row(self):
        """The raw pixel row relative to the bottom of the chip. All layers orient rows in the opposite direction to phi."""
        return self.hitIndex & 0xFF
    def x(self):
        """The pixel position along the beam line, taking account of chip orientation. See section 1.1.2 of the specbook."""
        if self.layer() < 3: return 256 - self.column()
        else: return self.column()
    def y(self):
        """The pixel position around phi, taking account of chip orientation so that increasing y moves with increasing phi."""
        return 250 - self.row()
    def station(self):
        """The station the chip is in. 0 = central; 1 = upstream; 2 = downstream."""
        return int(self.chipid() / (0x1 << 12))
    def layer(self): return int((self.chipid() / (0x1 << 10)) % 4 + 1)
    def phi(self):
        """How far around in phi the chip is. Essentially which ladder the chip is on."""
        return int((self.chipid() / (0x1 << 5)) % (1 << 5) + 1)
    def z(self):
        """How far along the ladder the chip is. Higher number is further downstream."""
        zt = self.chipid() % (1<<5);
        if self.layer() == 3:
            return zt - 7
        elif self.layer() == 4:
            return zt - 6;
        else: return zt;

def GlobalCalculator(v,drow,dcol, row_number, column_number):
    """Function that makes the final 3d coordinates of a hit from the """

    hit_global_coordinates = v + drow * (0.5 + row_number) + dcol * (0.5 + column_number)
    return hit_global_coordinates

def preprocess_chip_id_mapping(sensor_tree):
    """Precompute sensor data (v, drow, dcol) and store in a dictionary for fast lookup."""
    id_branch = sensor_tree["sensor"].array()
    branches = ["vx", "vy", "vz", "rowx", "rowy", "rowz", "colx", "coly", "colz"] # Branches that we need
    arrays = sensor_tree.arrays(branches)

    sensor_data_dict = {}
    for idx, chip_id in enumerate(id_branch): #Way of linking the chip id to the index in the sensor branch, 
        v = np.array([arrays["vx"][idx], arrays["vy"][idx], arrays["vz"][idx]])
        drow = np.array([arrays["rowx"][idx], arrays["rowy"][idx], arrays["rowz"][idx]])
        dcol = np.array([arrays["colx"][idx], arrays["coly"][idx], arrays["colz"][idx]])

        sensor_data_dict[chip_id] = (v, drow, dcol)  # Store sensor data
    return sensor_data_dict

def HitsInFrame(frame_number, mu3eTree, sensor_data_dict, mchits_data):
    """ Takes the input root file, and the frame number, and outputs an array of the hit information for that frame
    Inputs: mu3eTree: An array of all frames and the pixelIDs, frame number
    Outputs: Array of hit info for frame number
    """
    mu3eFrame = ak.Array([mu3eTree[frame_number]])
    #print(mu3eFrame)
    frame_hits =[] # Initialize list for storing hit information
    for hitsInFrame, timestamps, mc_indexes, mc_numbers in zip(mu3eFrame['hit_pixelid'], mu3eFrame['hit_timestamp'], mu3eFrame['hit_mc_i'], mu3eFrame['hit_mc_n']): # Loop for iterating through frame hits
        for hitIndex, time, mcIndex , mcNumber in zip(hitsInFrame, timestamps, mc_indexes, mc_numbers):
            hit = Hit(hitIndex)
            sensor_id = hit.chipid()
            row = hit.row()
            column = hit.column()

            # Lookup precomputed sensor data 
            sensor_info = sensor_data_dict.get(sensor_id)
            v, drow, dcol = sensor_info 
            hit_global_coords = v + drow * (0.5 + row) + dcol * (0.5 + column)

            mc_hit_info = mchits_data.get(mcIndex, {"tid", "hid", "hid_g"}) #Extract rel mc info based on index 
            frame_hits.append({ 
                'frameNumber': frame_number,           # Append a dictionary with the hit information
                'hitIndex': hit.hitIndex, # Not sure actually need the hit index? Include for now - actually useful for denoting each one
                'sensor_id': sensor_id,
                'row': row,
                'column': column,
                'station': hit.station(),
                'layer': hit.layer(),
                'ladder': hit.phi(),
                'chip': hit.z(),
                'pixel_x': hit.x(),
                'pixel_y': hit.y(),
                'timestamp' :time,
                'mcIndex': mcIndex,
                'mcNumber': mcNumber,
                'tid': mc_hit_info["tid"],
                'hid': mc_hit_info["hid"],
                'hid_g': mc_hit_info["hid_g"],
                'gx': hit_global_coords[0],
                'gy': hit_global_coords[1],
                'gz': hit_global_coords[2],
            })
        break
    return frame_hits # Return awkward array with the frame hits data

def CompileHits(signal_file, signal_no):
    """Function for compiling the hits_data"""

    # Inputting a file and state which one testing
    #print("Test with the file:", sort_file) 
    #print()

    # Open the root file, access the hits tree, find the total number of frames.
    mu3eTree = signal_file['mu3e'].arrays(['hit_pixelid', 'hit_timestamp', 'hit_mc_i', 'hit_mc_n']) # Opens just the Mu3eTree branches we need

    #Open the sensors tree, open the id branch, create the lookup dictionary mapping sensor_ids to indexes (Needed for global coordinates)
    sensor_tree = signal_file["alignment/sensors;1"] 
    id_branch = sensor_tree["sensor"].array()  
    sensor_id_to_index = preprocess_chip_id_mapping(sensor_tree) 

    #Make the lookup dict linking sensor chip id to correct v,drow,dcol info.
    sensor_data_dict = preprocess_chip_id_mapping(signal_file["alignment/sensors;1"])

    #Open the mchits tree, make a lookup dict preserving index no.
    mchits = signal_file["mu3e_mchits"].arrays(["tid", "hid", "hid_g"])
    print("Building mchits dictionary: Takes around 2 mins")
    mchits_data = mchits_data = {i: {"tid": tid, "hid": hid, "hid_g": hid_g}        # This is necessary as the tid are stored
                for i, (tid, hid, hid_g) in tqdm(enumerate(zip(mchits["tid"], mchits["hid"], mchits["hid_g"])), total=len(mchits["tid"]), desc="Building mchits dictionary", ncols=100)}

    #total frames and numbers:
    total_frames = len(mu3eTree)
    frame_numbers = list(range(0, total_frames)) 
    #print('Total number of frames in file:',total_frames)

    #Iterate over all frames, collect hit information
    all_hits = [] # List for all the frame hits to be appended to  
    for frame_number in tqdm(frame_numbers, desc="Processing frames", ncols=100):
        #print("Frame number:", frame_number) #NOTE: Means lots of print statements. Remove when confident, but good way to track progess
        frame_hits = HitsInFrame(frame_number, mu3eTree,sensor_data_dict, mchits_data)
        all_hits.extend(frame_hits)

    # Sort hits by frame number and then by tid within each frame
    #all_hits.sort(key=lambda x: (x['frameNumber'], x['tid'],x['hid'])) #NOTE: If this line is included, you sort training data into tracks anyway!! Could be an issue.


    # Converting array to Panda
    hit_data = pd.DataFrame(all_hits)
    
    #If want to save as csv
    # file_name = f"{signal_no}_hits_data.csv"
    # hit_data.to_csv(os.path.join(directory, file_name), index=False) #NOTE: Might be better to have different format, but CSV fine for now

    return hit_data

# Truth data extraction
def TruthInfo(frame_number, mu3eTree):
    """Find truth info for a single frame."""
    frame_mc_data = ak.zip({
        'frameNumber': frame_number,
        'hit_in_frame': ak.local_index(mu3eTree['traj_ID'][frame_number]),  
        'traj_ID': mu3eTree['traj_ID'][frame_number],
        'traj_mother': mu3eTree['traj_mother'][frame_number],
        'traj_PID': mu3eTree['traj_PID'][frame_number],
        'traj_type': mu3eTree['traj_type'][frame_number],
        'traj_time': mu3eTree['traj_time'][frame_number],
        'traj_vx': mu3eTree['traj_vx'][frame_number],
        'traj_vy': mu3eTree['traj_vy'][frame_number],
        'traj_vz': mu3eTree['traj_vz'][frame_number],
        'traj_px': mu3eTree["traj_px"][frame_number], #Momentum vector of the particle at creation'
        'traj_py': mu3eTree["traj_py"][frame_number], #Momentum vector of the particle at creation'
        'traj_pz': mu3eTree["traj_pz"][frame_number], #Momentum vector of the particle at creation'

	    'traj_fbhid': mu3eTree["traj_fbhid"][frame_number], #number of SciFi crossings
	    'traj_tlhid': mu3eTree["traj_tlhid"][frame_number], #number of passages through tile volume  
        'traj_edep_target': mu3eTree["traj_edep_target"][frame_number], #Energy deposited in the target by this particle
    })
    return ak.to_list(frame_mc_data)  

def CompileTruth(signal_file, signal_no):
    """Function for compiling the truth info into a panda"""
    mu3eTree = signal_file['mu3e'].arrays([
        'Ntrajectories', 'traj_ID', 'traj_mother', 'traj_PID', 'traj_type', 
        'traj_time', 'traj_vx','traj_vy','traj_vz','traj_px','traj_py','traj_pz', 'traj_fbhid', 'traj_tlhid', 'traj_edep_target'
    ]) # Opens just the Mu3eTree branches we need

    total_frames = len(mu3eTree)
    frame_numbers = list(range(0, total_frames)) # Set to 5000 for now to avoid files being too big for git pushing
    #print('Total number of frames in file:',total_frames)

    #Iterate over all frames, collect hit information
    all_hits = [] # List for all the frame hits to be appended to  
    for frame_number in tqdm(frame_numbers, desc="Processing frames", ncols=100):
        #print("Frame number:", frame_number)
        frame_hits = TruthInfo(frame_number, mu3eTree)
        all_hits.extend(frame_hits)
        
    # Converting array to Panda
    truth_data = pd.DataFrame(all_hits)
    
    #If want to save as csv
    # file_name = f"{signal_no}_truth_data.csv" 
    # truth_data.to_csv(os.path.join(directory, file_name), index=False) #NOTE: Might be better to have different format, but CSV fine for now

    return truth_data

# Now putting these together:
def LinkHitsTruthData(hits_data, truth_data, signal_no):
    """Function that takes the relevant data files for hits, and their ground truth values, and builds a single panda frame with this linked
    Input:
    - hits_data: Panda of raw hits data and their tids (the true track they belong to)
    - truth_data: Panda of the tid track truth info (from the _sort file mc info: denoted traj_... )

    Output:
    - Panda DataFrame containing Info on hits, ground truth momenta of tracks
    """
    # Count no. of each tid in hit data, filter out all with <4 hits (our condition 'reconstructable') 
    # ##NOTE: Not sure if correct to do this here, actually not going to do for now.
    # hits_data['num_hits'] = hits_data.groupby('tid')['tid'].transform('count')
    # hits_data_4_hits = hits_data[hits_data['num_hits'] >= 4]

    #NOTE:Temporary: Drop the columns not interested in. Update functions above to not include these
    hits_data.drop(columns =['station','column','layer','ladder','chip','pixel_x', 'pixel_y', 'timestamp', 'mcIndex','mcNumber','hid_g'], inplace = True)

    #Drop all the duplicate 'traj_tid' entries: These are all the same:
    truth_data_unique = truth_data.drop_duplicates(subset=['traj_ID'])

    # Merge hits data with truth file for each hit, save this as a raw csv
    hits_truth_merge = hits_data.merge(
        truth_data_unique[['traj_ID','traj_type','traj_px', 'traj_py', 'traj_pz']],
        left_on='tid',
        right_on='traj_ID',
        how='left'
    )
    hits_truth_merge.drop(columns=['traj_ID'], inplace=True)
    hits_truth_merge.dropna(subset = ['traj_px'], inplace = True) #Do this as some hits that do not have any truth info. We discard these hits.
    hits_truth_merge = hits_truth_merge[hits_truth_merge['traj_type'] != 0] #Remove any rows where traj_type = 0. These are photons
                                                                            #Not the most physical thing in the world but they cause problems later and I cannot deal with them now

    # # Compute truth track info: total p, pt, phi and lambda angle.(Needed for binning tracks later)
    hits_truth_merge['traj_p'] = np.sqrt(
        hits_truth_merge['traj_px']**2 + 
        hits_truth_merge['traj_py']**2 + 
        hits_truth_merge['traj_pz']**2
    )
    hits_truth_merge['traj_pt'] = np.sqrt(
        hits_truth_merge['traj_px']**2 + 
        hits_truth_merge['traj_py']**2
    )
    hits_truth_merge['traj_lambda'] = np.arctan2(hits_truth_merge['traj_pz'], hits_truth_merge['traj_pt'])
    hits_truth_merge['traj_phi'] = np.arctan2(hits_truth_merge['traj_py'], hits_truth_merge['traj_px'])

    print(f"length of hits data:{len(hits_data)}")
    print(f"length of merged data:{len(hits_truth_merge)} (Will be lower as some hits no traj_ID)")

    # file_path = f"/root/Mu3eProject/RawData/TransformerData/{signal_no}/{signal_no}_hits_truth_merge.csv"
    # hits_truth_merge.to_csv(file_path, index=False)
    return hits_truth_merge

def IndexedData (hits_truth_merge, momentum_bins, lambda_bins, phi_bins, q_mapping, signal_no):
    """Function for binning the hits data into bins based on the underlying truth values of the track, and returning the bin index.
    Inputs:
    - hits_truth_merge: Dataframe of hits data with truth info
    - ___bins: Bin sizes for data
    Output:
    Hits data with index"""

    # Create bin indexes
    hits_truth_merge['p_bin'] = np.digitize(hits_truth_merge['traj_p'], bins=momentum_bins, right=False) - 1
    hits_truth_merge['lambda_bin'] = np.digitize(hits_truth_merge['traj_lambda'], bins=lambda_bins, right=False) - 1
    hits_truth_merge['phi_bin'] = np.digitize(hits_truth_merge['traj_phi'], bins=phi_bins, right=False) - 1
    hits_truth_merge['type_bin'] = hits_truth_merge['traj_type'].map(q_mapping)
    hits_truth_merge.dropna(subset=['type_bin'], inplace=True) #Note this is a litte unphysical - dropping every hit except those defined below from the processes expect!
                                                               #However as a first test of getting the hits working for all files, it'll do. I leave it to the people after me to make a more thorough grouping.


    # Assign unique bin index for classification. Idea here is defining on each bin type by keeping in blocks of numbers
    hits_truth_merge['bin_index'] = (hits_truth_merge['p_bin'] * (num_lam_bins * num_phi_bins * 2) +  # 30 lambda bins, 30 phi bins, 2 type bins
                       hits_truth_merge['lambda_bin'] * (num_phi_bins * 2) +
                       hits_truth_merge['phi_bin'] * 2 +
                       hits_truth_merge['type_bin']) 

    hits_truth_bin_indexed = hits_truth_merge #Renaming to keep track that here has now been indexed

    #Save this as the master truth csv, for all hits (i.e before you split to training/test sets for the model)
    # file_path = f"{directory}{signal_no}_all_hits_truth_master.csv"
    # hits_truth_merge.to_csv(file_path, index=False)

    # If you want filtered data:
    # hits_truth_filtered = hits_truth_merge[["frameNumber","tid","gx","gy","gz","bin_index"]]
    # file_path = f"{directory}{signal_no}_hits_truth_indexed_filtered.csv"
    #hits_truth_filtered.to_csv(file_path, index=False)
    return hits_truth_bin_indexed


# Making tensor files:
class FrameDataset(Dataset):
    def __init__(self, data):
        # Load your CSV data
        self.df = data
        # Group by the frameNumber column
        self.grouped = self.df.groupby("frameNumber")
        # Get a sorted list of frame numbers
        self.frames = sorted(self.grouped.groups.keys())
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame_num = self.frames[idx]
        # Get all rows for this frame
        frame_data = self.grouped.get_group(frame_num)
        # Extract features (gx, gy, gz)
        features = torch.tensor(frame_data[['gx','gy','gz']].values, dtype=torch.float32)
        # Extract labels (bin_index)
        labels = torch.tensor(frame_data['bin_index'].values, dtype=torch.long)
        return features, labels, frame_num

    
def save_tensors_and_csv(features, labels, frame_nums, save_path, file_prefix):
    """Save features and labels as both .pt and .csv files."""
    tensor_path = os.path.join(save_path, f"{file_prefix}.pt")
    csv_path = os.path.join(save_path, f"{file_prefix}.csv")

    # Save as .pt
    torch.save((features, labels), tensor_path)

    # # Convert to DataFrame and save as CSV  - NOTE: This was used for testing tensors were saving correctly. Not really needed for training.
    # features_np = [f.numpy() for f in features]
    # labels_np = [l.numpy() for l in labels]
    # df = pd.DataFrame({
    #     "frameNumber": frame_nums,
    #     "features": [list(f.flatten()) for f in features_np],  # Flatten to save as CSV
    #     "labels": [list(l) for l in labels_np]
    # })
    # df.to_csv(csv_path, index=False)

    print(f"Saved {file_prefix} data to:\n  - {tensor_path}\n  - {csv_path}")


def DataToTorch(hits_truth_indexed, training_data_directory, signal_no, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=32):
    """Convert filtered, indexed hit data into train, val, and test datasets and save them as .pt and .csv files."""
    
    dataset = FrameDataset(hits_truth_indexed)
    all_frames = dataset.frames
    num_frames = len(all_frames)

    # Shuffle frame numbers
    random.seed(seed)
    frames_shuffled = all_frames.copy()
    random.shuffle(frames_shuffled)

    # Compute indices for splits
    train_end = int(train_ratio * num_frames)
    val_end = train_end + int(val_ratio * num_frames)
    
    train_frames = set(frames_shuffled[:train_end])
    val_frames = set(frames_shuffled[train_end:val_end])
    test_frames = set(frames_shuffled[val_end:])

    test_truths = hits_truth_indexed[hits_truth_indexed["frameNumber"].isin(test_frames)]
    os.makedirs(training_data_directory, exist_ok=True)

    # Save test truths CSV
    test_csv_path = os.path.join(training_data_directory, f"{signal_no}_test_truths_shuffled.csv")
    test_truths.to_csv(test_csv_path, index=False)
    print(f"Saved test helper CSV: {test_csv_path}")


    # # Save test helper as a pt and csv
# Save test_helper as a .pt and csv
    grouped_test_truths = test_truths.groupby("frameNumber")
    frames = sorted(grouped_test_truths.groups.keys())

    # Prepare a list of tuples (hit indices, event IDs)
    helper_per_frame = []
    csv_data = []  # Collect data for CSV
    for frame in frames:
        group = grouped_test_truths.get_group(frame)
        hit_indices = group["hitIndex"].tolist()  # Extract hit indices
        event_ids = [frame] * len(hit_indices)  # Ensure event IDs are the same per frame    #NOTE THIS LINE MIGHT BE THE PROBLEM W THE TEST-HELPER AND COLLATE FUNCTION
        helper_per_frame.append((hit_indices, event_ids))
        
        # Collect for CSV
        for hit, event in zip(hit_indices, event_ids):
            csv_data.append((hit , event))

    # Save test_helper as .pt
    test_helper_path = os.path.join(training_data_directory, f"{signal_no}_test_helper.pt")
    torch.save(helper_per_frame, test_helper_path)
    print(f"Saved test helper .pt file: {test_helper_path}")

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(csv_data, columns=[ "hitIndex", "frameNumber"])
    csv_path = os.path.join(training_data_directory, f"{signal_no}_test_helper.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved test helper CSV: {csv_path}")


    # Containers for train, val, and test splits
    data_splits = {
        "train": ([], [], []),
        "val": ([], [], []),
        "test": ([], [], []),
    }

    # Loop over dataset and allocate frames
    for i in range(len(dataset)):
        features, labels, frame_num = dataset[i]
        if frame_num in train_frames:
            split = "train"
        elif frame_num in val_frames:
            split = "val"
        else:
            split = "test"
        
        data_splits[split][0].append(features)
        data_splits[split][1].append(labels)
        data_splits[split][2].append(frame_num)

    # Save all splits as .pt and .csv
    for split_name, (features, labels, frame_nums) in data_splits.items():
        save_tensors_and_csv(features, labels, frame_nums, training_data_directory, f"{signal_no}_{split_name}")

    print(f"Saved splits:\n  Train: {len(data_splits['train'][0])}\n  Validation: {len(data_splits['val'][0])}\n  Test: {len(data_splits['test'][0])}")


# Function for automating all above. This should really be in a separate file. I tried, and it wasn't working... you can make it more efficient if you want.
def ProcessRootFiles(root_dir, output_dir, p_bins, lam_bins, phi_bins, q_mapping):
    """Process all ROOT files in a given directory and generate training data."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_hit_data = []
    all_truth_data = []
    frame_offset = 0  # Keep track of frame numbers across files

    root_files = glob.glob(os.path.join(root_dir, "*.root"))

    print(f"Found {len(root_files)} ROOT files in {root_dir}")
    print(f"Found {len(root_files)} ROOT files: {root_files}")

    for root_file in root_files:
        # Extract the first two parts
        signal_no = "_".join(root_file.split("_")[:2])
        print(f"Processing {signal_no}")
        signal_file = uproot.open(root_file)
        print()
        # Compile hits and truth data
        print(f"Compiling hits data for {signal_no}")
        hit_data =  CompileHits(signal_file, signal_no)
        print(f"Compiling truth data for {signal_no}")
        truth_data =  CompileTruth(signal_file, signal_no)

        # Adjust frame numbers to avoid duplicates
        hit_data["frameNumber"] += frame_offset
        truth_data["frameNumber"] += frame_offset

        all_hit_data.append(hit_data)
        all_truth_data.append(truth_data)
        frame_offset += 9921  # Update offset for next file

    print("Merging data from all roots files:")
    merged_hits = pd.concat(all_hit_data, ignore_index=True) # Make one big dataframe of hits
    merged_truth = pd.concat(all_truth_data, ignore_index=True) #Make one big dataframe of truths

    # merged_hits.to_csv(f"{output_dir}merged_hits_check.csv", index=False)
    # merged_truth.to_csv(f"{output_dir}merged_truths_check.csv", index = False)

    # Process merged data
    print("Linking all hits and truth data:")
    merged_hits_truth =  LinkHitsTruthData(merged_hits, merged_truth, signal_no="merged")
    print("Data linked, beginning indexing:")
    all_bin_indexed_hits =  IndexedData(merged_hits_truth, p_bins, lam_bins, phi_bins, q_mapping, signal_no="merged")

    # Save merged CSV
    merged_csv_path = os.path.join(output_dir, "all_merged_truths_master.csv")
    all_bin_indexed_hits.to_csv(merged_csv_path, index=False)
    print(f"Saved all root file merged hits and Truth data: {merged_csv_path}")

    # Convert to PyTorch dataset

    print("Converting to Torch tensors")
    DataToTorch(all_bin_indexed_hits, output_dir, signal_no="merged")

    print("Processing complete")

# When making evaluation sets, compile all the trirec data:
def FrameTracks(frame_number, frames_tree):

    """Extract info on tracks and the corresponding mc info in a given frame of the trirec file.
    Input:
        frame_number
        frame_tree: The pre-loaded frame tree of the root file

    Returns:
        list: A list of dictionaries containing track information.
    """
    frames_frame = frames_tree[frame_number]  # Extract the specific frame
    frame_hits = []  # List to store track information

    # Extract track parameters for the frame
    frameId = frames_frame["frameId"]
    x0 = frames_frame["x0"]
    y0 = frames_frame["y0"]
    z0 = frames_frame["z0"]
    r = frames_frame["r"]
    p = frames_frame["p"]
    chi2 = frames_frame["chi2"]
    length = frames_frame["nhit"]
   # mc_eventId = frames_frame["mc_eventId"] 
    mc_prime = frames_frame["mc_prime"]
    mc = frames_frame["mc"]
    mc_tid= frames_frame["mc_tid"]
    mc_pid= frames_frame["mc_pid"] 
    mc_mid= frames_frame["mc_mid"]
    mc_type = frames_frame["mc_type"]
    mc_p = frames_frame["mc_p"]
    mc_pt = frames_frame["mc_pt"]
    mc_phi = frames_frame["mc_phi"]
    mc_lam = frames_frame["mc_lam"]
    mc_theta = frames_frame["mc_theta"]
    mc_vx = frames_frame["mc_vx"]
    mc_vy = frames_frame["mc_vy"]
    mc_vz = frames_frame["mc_vz"]
    mc_vr = frames_frame["mc_vr"]

    # Loop over tracks in this frame, making separate entries
    for i in range(len(frames_frame["x0"])):  
        frame_hits.append({
            'frameNumber': frame_number,  
            'frameId': frameId,  
            'x0': x0[i],  
            'y0': y0[i],  
            'z0': z0[i],  
            'r': r[i],  
            'p': p[i],  
            'chi2': chi2[i], 
            'length': length[i], 
            #'mc_eventId': mc_eventId[i],
            'mc_prime' : mc_prime[i],
            'mc measure' :mc[i],
            'mc_tid':mc_tid[i],
            'mc_pid':mc_pid[i],
            'mc_mid':mc_mid[i],
            'mc_type': mc_type[i],  
            'mc_p': mc_p[i],  
            'mc_pt': mc_pt[i],  
            'mc_phi': mc_phi[i],  
            'mc_lam': mc_lam[i],  
            'mc_theta': mc_theta[i],  
            'mc_vx': mc_vx[i],  
            'mc_vy': mc_vy[i],  
            'mc_vz': mc_vz[i],  
            'mc_vr': mc_vr[i]
        })

    return frame_hits

def CompileTrirec(trirec_file):
    """"Function that iterates over a whole trirec file to compile the tracks"""
    frames_tree = trirec_file['frames'].arrays()  # Load the frames tree as an awkward array
    total_frames = len(frames_tree)
    print('Total number of frames in file:', total_frames)
    frame_numbers = list(range(0, total_frames))

    # Iterate over all frames and collect track information
    all_hits = []
    for frame_number in tqdm(frame_numbers,desc="processing frames", ncols=100):
        frame_hits = FrameTracks(frame_number, frames_tree)
        all_hits.extend(frame_hits)

    # Convert to Pandas DataFrame
    all_frames_data = pd.DataFrame(all_hits)
    
    return all_frames_data

def ProcessTrirecFiles(root_dir, output_dir):
    """Process all ROOT files in a given directory and generate merged data of all trirec info."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_trirec_data = []
    frame_offset = 0  # Keep track of frame numbers across files

    root_files = glob.glob(os.path.join(root_dir, "*.root"))

    print(f"Found {len(root_files)} ROOT files in {root_dir}")
    print(f"Found {len(root_files)} ROOT files: {root_files}")

    for root_file in root_files:
        # Extract the first two parts
        signal_no = "_".join(root_file.split("_")[:2])
        print(f"Processing {signal_no}")
        signal_file = uproot.open(root_file)
        print()

        # Compile hits and truth data
        print(f"Compiling trirec data for {signal_no}")
        trirec_data =  CompileTrirec(signal_file)


        # Adjust frame numbers to avoid duplicates
        trirec_data["frameNumber"] += frame_offset

        all_trirec_data.append(trirec_data)
        frame_offset += 9921  # Update offset for next file

    print("Merging data from all roots files:")
    merged_trirec = pd.concat(all_trirec_data, ignore_index=True) # Make one big dataframe of hits

    merged_trirec.to_csv(f"{output_dir}merged_trirec_data_ALL.csv", index=False)
    print("Saved all merged trirec data ")

    return merged_trirec

# Make the comparison data for the current alg
def BuildComparisonData(merged_hits_truth_file, trirec_data, output_dir):
    """Function that takes the relevant data files for hits with their ground truth values, and assigned tracks,   
    and builds a single data file for evaluation.
    Input:
    - merged_hits_truth_file: File containing raw hits data and its truth params, made using the "compile_datasets.py" code
    - trirec_file: File containing trirec reconstruction data from current algorithm, and its truth parameters. 
    #NOTE: If function changed to work for output of the ML model, will need to change trirec_file to be OUTPUT FILE OF ML MODEL.
    Output:
    - Single file containing Info on no. of unique 4-hit reconstructable tracks, ground truth momenta of tracks, and reconstruction attempt
    """

    #Open the 2 different files and read the csvs
    merged_hits_truth_data = pd.read_csv(merged_hits_truth_file)

    
    # Count no. of each tid in hit data, remove duplicates, filter out all with <4 hits (our condition 'reconstructable')
    merged_hits_truth_data['num_hits'] = merged_hits_truth_data.groupby('tid')['tid'].transform('count')
    filtered_hits_truth_data = merged_hits_truth_data[merged_hits_truth_data['num_hits'] >= 4].drop_duplicates(subset=['tid'])



    #Count no. of reconstructed tracks for single tid in trirec file. Then remove duplicates (SEE NOTE)
    trirec_data['num_recon_tracks'] = trirec_data.groupby('mc_tid')['mc_tid'].transform('count')
  #trirec_data.drop_duplicates(subset=['mc_tid']) 
    # NOTE:This line removes duplicates of mc_tid. You want to REMOVE duplicates for the overall efficiency, but KEEP duplicates for the fake rate! 
    # For now: I simply keep the duplicates here, then remove them in the function for overall efficiency


    # Merge with trirec data 
    all_merged = filtered_hits_truth_data.merge(
        trirec_data[['length', 'mc_prime', 'mc measure', 'mc_tid', 'num_recon_tracks', 'mc_vx', 'mc_vy', 'mc_vz', 'mc_p','mc_pt', 'mc_lam']],  # Keep only relevant columns
        left_on='tid',
        right_on='mc_tid',
        how='left'  # Keeps all rows from merged_df, fills missing trirec info with NaN
    )

    #df_merged_truths = all_merged.dropna(subset = ['traj_px'])

    # # Print results
    # print("Final dataframe with merged info:")
    # print(df_merged_truths.head())  # Print first few rows to check

    comparison_data_path = os.path.join(output_dir, "current_alg_comparison_data.csv")
    all_merged.to_csv(comparison_data_path, index=False)
    return 


#########################################################################################################################################
##### Define the bins for indexing NOTE: Ideally this will be turned into a TOML self contained dict or something 
p_min = 0
p_max = 65
num_p_bins = 3
p_bins = np.linspace(p_min,p_max,num_p_bins+1)

lam_min = -1.6
lam_max = 1.6
num_lam_bins = 30
lam_bins = np.linspace(lam_min,lam_max,num_lam_bins+1)

phi_min = -np.pi
phi_max = np.pi
num_phi_bins = 30
phi_bins = np.linspace(phi_min,phi_max,num_phi_bins+1)

q_mapping = {11:0,  #Mapping to charges for: positrons, electrons (from michel, bhabha, signal)
             91:0,  #Positron charge -> 0, Electron charge ->1
             21:0,
             31:0,
             41:0, #See wiki naming conventions for what these are. All positron/electron charge. Photons dropped earlier (sorry photons)
             3:0,  # Actually, anything bar these types ends up being dropped. This is not very physical.
             32:1, # But, I needed a quick way to remove all the NaN values as they mess with the tensors. I have highlighted above where all NaN are removed.
             42:1, # Ideally a more complete dictionary of event possibilites will be created and used. 
             52:1,
             82:1,
             92:1,}  
                    
total_bins = num_p_bins * num_lam_bins *num_phi_bins * 2
#print(f"Total number of bins: {total_bins}")


################################## Hits Data Conversion and CSV making ########################################################

root_dir = "/root/Mu3eProject/DataFilesAndTests/DataAutomationTest/RootFiles/"
output_dir = "/root/Mu3eProject/DataFilesAndTests/DataAutomationTest/TestSet4/Evaluation/"
trirec_dir = "/root/Mu3eProject/DataFilesAndTests/DataAutomationTest/TrirecFiles/"

ProcessRootFiles(root_dir, output_dir, p_bins, lam_bins, phi_bins, q_mapping)

## When you are going to compile evaluation, and test current algorithm:
merged_trirec = ProcessTrirecFiles(trirec_dir, output_dir) #NOTE: Run this when you need to compile the extracting 
merged_hits_truth_data = "/root/Mu3eProject/DataFilesAndTests/DataAutomationTest/TestSet4/Evaluation/merged_test_truths_shuffled.csv" 
#NOTE: make sure the path for merged_hits_truth_data is that where the ProcessRootFiles will save the eval merged_hits_truth_data file.

BuildComparisonData(merged_hits_truth_data, merged_trirec, output_dir)


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Process multiple ROOT files into training data.")
#     parser.add_argument("root_dir", help="Directory containing ROOT files")
#     parser.add_argument("output_dir", help="Directory to save processed data")
#     args = parser.parse_args()
    
#     process_root_files(args.root_dir, args.output_dir)




#### Below if you want to run for one training file separately. (This is old code before automated)
# # Open file for conversion
# signal_no = "signal1_97"
# sort_file = "/root/Mu3eProject/RawData/SortDataFilesV5.3/signal1_97_32652_execution_1_run_num_716703_sort.root" #Not worked out how to get the last bit to manually adjust
# signal_file = uproot.open(sort_file) # Opens the file


# # Create directory for file saving
# directory = f"/root/Mu3eProject/RawData/TransformerData/TestUnsorted/{signal_no}/" #NOTE: currently this needs to be changed each time
# if not os.path.exists(directory):
#     os.makedirs(directory)

# #Compile hits data, truth data, merge, index.
# print("Compiling hits data:")
# print()
# hit_data = CompileHits(signal_file, signal_no)
# print("Hits data compiled.")
# print("Compiling truth data:")
# truth_data = CompileTruth(signal_file, signal_no)
# print("Hits data compiled")
# print("Merging data:")
# hits_truth_merge = LinkHitsTruthData(hit_data,truth_data, signal_no = signal_no)
# print("Data merged, beginning indexing:")

# hits_truth_bin_indexed = IndexedData(hits_truth_merge,directory, p_bins,lam_bins,phi_bins,q_mapping, signal_no = signal_no)
# print("Data indexed, master truth file")

# ############## Turn Hits Data CSVs into Torch tensor files, and csv helper files ########################################

# helper_test_file = f"/root/Mu3eProject/RawData/TransformerData/TestUnsorted/{signal_no}/{signal_no}_all_hits_truth_master.csv"


# training_data_directory = f"/root/Mu3eProject/RawData/TransformerData/TestUnsorted/{signal_no}"
# hits_truth_indexed = pd.read_csv(helper_test_file) #For loading in the merged data for the helper file. Need to edit this to make neater.
#                                                 # Should not be calling in so many functions to this below
#                                                 # Should ideally be working with the panda output of th other code so not re-opening stuff

# DataToTorch(hits_truth_indexed,training_data_directory, signal_no = signal_no)




