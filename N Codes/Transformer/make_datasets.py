import uproot
import pandas as pd
import numpy as np
import awkward as ak
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random

###### Use this code to create the data sets for the transformer model. Saves all the files into a data_training folder.
# Make this code just by editing the extract_hits optimised

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
    print("Test with the file:", sort_file) 
    print()

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
    mchits_data = mchits_data = {i: {"tid": tid, "hid": hid, "hid_g": hid_g} 
                for i, (tid, hid, hid_g) in enumerate(zip(mchits["tid"], mchits["hid"], mchits["hid_g"]))}

    #total frames and numbers:
    total_frames = len(mu3eTree)
    frame_numbers = list(range(0, total_frames)) 
    print('Total number of frames in file:',total_frames)

    #Iterate over all frames, collect hit information
    all_hits = [] # List for all the frame hits to be appended to  
    for frame_number in frame_numbers:
        print("Frame number:", frame_number) #NOTE: Means lots of print statements. Remove when confident, but good way to track progess
        frame_hits = HitsInFrame(frame_number, mu3eTree,sensor_data_dict, mchits_data)
        all_hits.extend(frame_hits)

    # Sort hits by frame number and then by tid within each frame
    all_hits.sort(key=lambda x: (x['frameNumber'], x['tid'],x['hid']))


    # Converting array to Panda, and then saving as CSV 
    hit_data = pd.DataFrame(all_hits)#
    file_name = f"{signal_no}_hits_data.csv"
    hit_data.to_csv(os.path.join(directory, file_name), index=False) #NOTE: Might be better to have different format, but CSV fine for now

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
    print('Total number of frames in file:',total_frames)

    #Iterate over all frames, collect hit information
    all_hits = [] # List for all the frame hits to be appended to  
    for frame_number in frame_numbers:
        print("Frame number:", frame_number)
        frame_hits = TruthInfo(frame_number, mu3eTree)
        all_hits.extend(frame_hits)
        

    # Converting array to Panda, and then saving as CSV 
    truth_data = pd.DataFrame(all_hits)
    file_name = f"{signal_no}_truth_data.csv"
    truth_data.to_csv(os.path.join(directory, file_name), index=False) #NOTE: Might be better to have different format, but CSV fine for now

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

    # Assign unique bin index for classification. Idea here is defining on each bin type by keeping in blocks of numbers
    hits_truth_merge['bin_index'] = (hits_truth_merge['p_bin'] * (num_lam_bins * num_phi_bins * 2) +  # 30 lambda bins, 30 phi bins, 2 type bins
                       hits_truth_merge['lambda_bin'] * (num_phi_bins * 2) +
                       hits_truth_merge['phi_bin'] * 2 +
                       hits_truth_merge['type_bin']) 

    file_path = f"/root/Mu3eProject/RawData/TransformerData/{signal_no}/{signal_no}_hits_truth_indexed.csv"
    hits_truth_merge.to_csv(file_path, index=False)

    hits_truth_filtered = hits_truth_merge[["frameNumber","tid","gx","gy","gz","bin_index"]]
    file_path = f"/root/Mu3eProject/RawData/TransformerData/{signal_no}/{signal_no}_hits_truth_indexed_filtered.csv"
    hits_truth_filtered.to_csv(file_path, index=False)
    return hits_truth_filtered



class FrameDataset(Dataset):
    def __init__(self, csv_file):
        # Load your CSV data
        self.df = pd.read_csv(csv_file)
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
        return features, labels
    
    
def DataToTorch(hits_truth_filtered,training_data_directory,signal_no):
    """Function for taking the filtered, indexed hits and truth data, and turning it into data files suitable for loading into the transformer
    Input:
    -hits_truth_filtered: Panda DF of frames, hits data 3d coordinates indexed track classification
    Output:
    -.pt model training files"""

    dataset = FrameDataset(hits_truth_filtered)

    # Extract features, labels, and frame numbers
    features_list, labels_list = [], []

    for i in range(len(dataset)):
        features, labels = dataset[i]
        features_list.append(features)
        labels_list.append(labels)


    # Save as .pt file
    torch.save((features_list, labels_list), os.path.join(training_data_directory, "sorted_train.pt"))



##### Define the bins for indexing NOTE: Ideally this will be turned into a TOML self contained dict or something...
p_min = 0
p_max = 65
num_p_bins = 3
p_bins = np.linspace(p_min,p_max,num_p_bins+1)
print(f"pbins: {p_bins}")

lam_min = -1.6
lam_max = 1.6
num_lam_bins = 30
lam_bins = np.linspace(lam_min,lam_max,num_lam_bins+1)

phi_min = -np.pi
phi_max = np.pi
num_phi_bins = 30
phi_bins = np.linspace(phi_min,phi_max,num_phi_bins+1)

q_mapping = {11: 0, 91: 0 ,52: 1, 92: 1}  #Mapping to charges for: positrons, electrons (from michel, bhabha, signal)
                                          #Positrons -> 0, Electrons ->1

total_bins = num_p_bins * num_lam_bins *num_phi_bins * 2
print(f"Total number of bins: {total_bins}")

############################################################################################################

# Open file for conversion
sort_file = "/root/Mu3eProject/RawData/SortDataFilesV5.3/signal1_96_32652_execution_1_run_num_789062_sort.root"
signal_file = uproot.open(sort_file) # Opens the file
signal_no = "signal1_96"

# Create directory for file saving
directory = f"/root/Mu3eProject/RawData/TransformerData/{signal_no}" #NOTE: currently this needs to be changed each time
if not os.path.exists(directory):
    os.makedirs(directory)

############################################################################################################
# print("Compiling hits data:")
# print()
# hit_data = CompileHits(signal_file, signal_no)
# print("Hits data saved.")
# print("Compiling truth data:")
# truth_data = CompileTruth(signal_file, signal_no)

# print("Merging data:")
# hits_truth_merge = LinkHitsTruthData(hit_data,truth_data, signal_no = signal_no)
# print("Data merged, beginning indexing:")

# IndexedData(hits_truth_merge,p_bins,lam_bins,phi_bins,q_mapping, signal_no = signal_no)
# print("Data indexed")

############## Testing turning into pytorch form: ########################
indexed_test_file = "/root/Mu3eProject/RawData/TransformerData/signal1_96/signal1_96_hits_truth_indexed_filtered.csv"
# Example usage:

training_data_directory = "/root/Mu3eProject/RawData/TransformerData/TrainingData"
if not os.path.exists(training_data_directory):
    os.makedirs(training_data_directory)


DataToTorch(indexed_test_file,training_data_directory, signal_no)


# Load the .pt file
data = torch.load("/root/Mu3eProject/RawData/TransformerData/TrainingData/sorted_train.pt")

# Check the type of data stored
print(type(data))

# Check the length of the tuple
print(f"Tuple length: {len(data)}")

# Inspect the first few elements
for i, item in enumerate(data):
    print(f"Element {i}: Type: {type(item)}")

    # If it's a tensor, show its shape
    if isinstance(item, torch.Tensor):
        print(f"  Shape: {item.shape}")
        print(f"  First few values: {item[:5]}")
    
    # If it's a list or dict, show some details
    elif isinstance(item, list):
        print(f"  List length: {len(item)}")
        print(f"  First item: {item[0]}")
    elif isinstance(item, dict):
        print(f"  Dictionary keys: {item.keys()}")
    
    # Only print details for the first few items to avoid too much output
    if i >= 2:
        break
