import uproot
import awkward as ak
import pandas as pd 
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

def HitsInFrame(filename, frame_number):
    """ Function that takes the file, and the frame number, and outputs a panda of the hit information for that frame"""
    inputFile = uproot.open(filename)
    mu3eTree = inputFile['mu3e'].arrays()
    mu3eFrame = ak.Array([mu3eTree[frame_number]])
    #print(mu3eFrame)
    frame_hits =[] # Initialize list for storing hit information
    for hitsInFrame in mu3eFrame['hit_pixelid']: # Loop for iterating through frame hits
        for hitIndex in hitsInFrame:
            hit = Hit(hitIndex)
            frame_hits.append({ 
                'frameNumber': frame_number,           # Append a dictionary with the hit information
                'hitIndex': hit.hitIndex, # Not sure actually need the hit index? Include for now - actually useful for denoting each one
                'station': hit.station(),
                'layer': hit.layer(),
                'ladder': hit.phi(),
                'chip': hit.z(),
                'pixel_x': hit.x(),
                'pixel_y': hit.y()
                })
        break

    hits_data_frame = pd.DataFrame(frame_hits)
    #print(hits_data_frame)  # Display the DataFrame for verification

    return hits_data_frame  # Return the DataFrame for further use
    #print(frame_hits)

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) == 1:
#         print("You need to specify an input file(s)")
#         sys.exit(-1)

    # for argument in sys.argv[1:]:
    #     print("File:", argument)
    #     printHitsInFirstFrame(argument)

#######################################################################################################
# Inputting a file and testing the output
file_path = "/root/Mu3eProject/RawData/v5.3/signal1_96_32652_execution_1_run_num_789062_sort.root"
print("Test with the file:", file_path) 
print()

#Find number of frames have:
signal_file = uproot.open(file_path)
frames = signal_file['mu3e'].arrays()
total_frames = len(frames)
print('Total number of frames in file:',total_frames)
print()

# Create directory for file using
directory = "/root/Mu3eProject/WorkingVersion/Mu3eProject/v5.3/signal1_96_32652" # Directory : NOTE currently this needs to be changed each time
#directory = "/root/Mu3eProject/WorkingVersion/signal1_1_1_all"
if not os.path.exists(directory):
    os.makedirs(directory)

file_name = "hits_data_signal1_96_32652_old_code.csv"
if os.path.exists(file_name): # Deletes old version of file if present
    os.remove(file_name)

#Iterate over all frames in file
frame_numbers = list(range(0, total_frames + 1)) 

#Saving frame hit information as a csv
for frame_number in frame_numbers:
    print("Frame number:", frame_number)
    print()
    frame_hits_data = HitsInFrame(file_path, frame_number) # Use the function to output the panda for frame hit information

    if frame_number ==0:
         frame_hits_data.to_csv(os.path.join(directory, file_name), index=False, mode ='w') # Keeping this here for now whilst testing 

    else:
        frame_hits_data.to_csv(os.path.join(directory,file_name), index=False, mode='a', header=False)

    #print(frame_hits_data)


