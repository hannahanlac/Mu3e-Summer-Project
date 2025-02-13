import uproot
import awkward as ak
import pandas as pd
import os

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


# Create directory for file saving
directory = "/root/Mu3eProject/RawData/TrirecFiles/signal1_99_32652"
if not os.path.exists(directory):
    os.makedirs(directory)

file_name = "trirec_data_signal1_99_32652_frames.csv"
file_path = os.path.join(directory, file_name)

if os.path.exists(file_path):  # Deletes old version of file if present
    os.remove(file_path)

# Load the ROOT file and extract the frames tree
root_file_path = "/root/Mu3eProject/RawData/TrirecFiles/signal1_99_32652_execution_1_run_num_561343_trirec.root"
print("Processing file:", root_file_path)

trirec_file = uproot.open(root_file_path)  # Open the ROOT file
frames_tree = trirec_file['frames'].arrays()  # Load the frames tree as an awkward array

total_frames = len(frames_tree)
print('Total number of frames in file:', total_frames)

# Iterate over all frames and collect track information
all_hits = []
for frame_number in range(total_frames):
    print(f"Frame {frame_number}")
    frame_hits = FrameTracks(frame_number, frames_tree)
    all_hits.extend(frame_hits)

# Convert to Pandas DataFrame and save to CSV
all_frames_data = pd.DataFrame(all_hits)
all_frames_data.to_csv(file_path, index=False)









