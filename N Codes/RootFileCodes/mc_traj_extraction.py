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
    return ak.to_list(frame_mc_data)  # Convert to list for further processing

#######################################################################################################

# Create directory for file saving
directory = "/root/Mu3eProject/RawData/TransformerData/signal1_98" #NOTE: currently this needs to be changed each time
if not os.path.exists(directory):
    os.makedirs(directory)
file_name = "signal1_98_32652_truth_data.csv"
if os.path.exists(file_name): # Deletes old version of file if present
    os.remove(file_name)

# Inputting a file and state which one testing
file_path = "/root/Mu3eProject/RawData/v5.3/signal1_98_32652_execution_1_run_num_135993_sort.root"
print("Test with the file:", file_path) 
print()

# Open the root file, access the hits tree, find the total number of frames.
signal_file = uproot.open(file_path) # Opens the file
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