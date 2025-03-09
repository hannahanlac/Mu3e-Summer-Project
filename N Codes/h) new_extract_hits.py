import uproot
import pandas as pd
import numpy as np
import awkward as ak
import os
import random
import glob
from tqdm import tqdm

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
    hits_data.drop(columns =['column', 'timestamp', 'mcIndex','mcNumber','hid_g'], inplace = True)

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


# Function for automating all above. This should really be in a separate file. I tried, and it wasn't working... you can make it more efficient if you want.
def ProcessRootFiles(root_dir, output_dir):
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
    merged_csv_path = os.path.join(output_dir, "new_hits_data_signal1_96_32652.csv")
    merged_hits_truth.to_csv(merged_csv_path, index = False)
    print(f"Saved all root file merged hits and Truth data: {merged_csv_path}")





################################## Hits Data Conversion and CSV making ########################################################

root_dir = "Simulation Data/nICK'S Special fOlder"
output_dir = "GithubRepoLinux/ProcessedData/signal1_96_32652/csv"



ProcessRootFiles(root_dir, output_dir)

