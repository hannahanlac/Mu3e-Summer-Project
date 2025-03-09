import uproot
import awkward as ak
import pandas as pd 
import numpy as np
import os

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


def HitsInFrame(frame_number, mu3eTrame):
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

#######################################################################################################

# Create directory for file saving
directory = "/root/Mu3eProject/RawData/TransformerData/signal1_98" #NOTE: currently this needs to be changed each time
if not os.path.exists(directory):
    os.makedirs(directory)
file_name = "hits_data_signal1_98_32652_with_global.csv"
if os.path.exists(file_name): # Deletes old version of file if present
    os.remove(file_name)

# Inputting a file and state which one testing
root_file_path = "/root/Mu3eProject/RawData/v5.3/signal1_98_32652_execution_1_run_num_135993_sort.root"
print("Test with the file:", root_file_path) 
print()

# Open the root file, access the hits tree, find the total number of frames.
signal_file = uproot.open(root_file_path) # Opens the file
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
    print("Frame number:", frame_number)
    frame_hits = HitsInFrame(frame_number, mu3eTree)
    all_hits.extend(frame_hits)

# Sort hits by frame number and then by tid within each frame
all_hits.sort(key=lambda x: (x['frameNumber'], x['tid'],x['hid']))


# Converting array to Panda, and then saving as CSV 
all_frames_data = pd.DataFrame(all_hits)
all_frames_data.to_csv(os.path.join(directory, file_name), index=False) #NOTE: Might be better to have different format, but CSV fine for now


# Need to fix this bit later 
# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) == 1:
#         print("You need to specify an input file(s)")
#         sys.exit(-1)

    # for argument in sys.argv[1:]:
    #     print("File:", argument)
    #     printHitsInFirstFrame(argument)