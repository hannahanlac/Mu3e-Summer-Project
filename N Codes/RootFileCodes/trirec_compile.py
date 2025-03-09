import uproot
import awkward as ak
import pandas as pd
import os
import glob
from tqdm import tqdm
import random
from torch.utils.data import Dataset

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
    """Process all ROOT files in a given directory and generate training data."""
    
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

# Bit useless
def TestTrirecSave (merged_trirec, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=32):
    """Function for saving specifically the frames used for testing the Transformer, to allow fair comparison"""

    dataset = FrameDataset(merged_trirec)
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

    test_frames = merged_trirec[merged_trirec["frameNumber"].isin(test_frames)]

    # Save test truths CSV
    test_csv_path = os.path.join(output_dir, "merged_test_truths_shuffled.csv")
    test_frames.to_csv(test_csv_path, index=False)
    print(f"Saved test helper CSV: {test_csv_path}")

### For making comparison data for




trirec_dir = "/root/Mu3eProject/DataFilesAndTests/DataAutomationTest/TrirecFiles"
output_dir = "/root/Mu3eProject/DataFilesAndTests/DataAutomationTest/OutputTest3/"

merged_trirec = ProcessTrirecFiles(trirec_dir, output_dir)

# TestTrirecSave(merged_trirec) NOTE: This created so shuffle in same was as test sets for transf. But as use eval sets, this not needed











