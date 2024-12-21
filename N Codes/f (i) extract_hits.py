import uproot
import awkward as ak
import pandas as pd 
import os
#os: Provides functions to interact with the operating system (e.g., file and directory operations)

class Hit(object):
    #snippet provided by Mark
    """Decodes a 32 bit hit ID (as found in the 'hit_pixelid' branch) into its constituent parts"""
    def __init__(self, hitIndex):
        self.hitIndex = hitIndex
    #Constructor that initializes the hitIndex, which contains the raw 32-bit hit information.
    
    def __str__(self):
        return "%d (station %d, layer %d, ladder %d, chip %d, pixel [%d, %d])" % (self.hitIndex, self.station(), self.layer(), self.phi(), self.z(), self.x(), self.y())
    #__str__: Returns a human-readable string representing the hit's details, such as station, layer, ladder, chip, and pixel coordinates (x, y).
    
    def chipid(self): return self.hitIndex >> 16
    def pixelid(self): return self.hitIndex
    #chipid: Extracts the chip ID by right-shifting (>>) the hitIndex by 16 bits.
    #pixelid: Returns the full 32-bit hitIndex.

    def column(self):
        """The raw pixel column relative to the left side of the chip. Different layers orient chips differently, so "left" is not consistent."""
        return (self.hitIndex >> 8) & 0xFF
    def row(self):
        """The raw pixel row relative to the bottom of the chip. All layers orient rows in the opposite direction to phi."""
        return self.hitIndex & 0xFF
    #column: Extracts the pixel's column by right-shifting by 8 bits and masking the lower 8 bits (& 0xFF).
    #row: Extracts the pixel's row by masking only the lower 8 bits of the hitIndex

    def x(self):
        """The pixel position along the beam line, taking account of chip orientation. See section 1.1.2 of the specbook."""
        if self.layer() < 3: return 256 - self.column()
        else: return self.column()
    def y(self):
        """The pixel position around phi, taking account of chip orientation so that increasing y moves with increasing phi."""
        return 250 - self.row()
    #x: Adjusts the pixel's position along the beamline (x) based on chip orientation:
    #For layers < 3, it flips the column (256 - column).
    #Otherwise, it keeps the raw column.
    #y: Adjusts the pixel's position in the phi direction by flipping the row (250 - row)


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
    #station: Determines the station number by dividing chipid by 2^12
    #layer: Determines the layer, using modulo arithmetic to isolate relevant bits.
    #phi: Determines the chip's ladder in the phi direction.
    #z: Determines the chip's position along the ladder with an adjustment for layers 3 and 4

def HitsInFrame(filename, frame_number):
    #snippet provided by Mark
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

#input: filename: Path to the ROOT file. frame_number: Frame to extract hits from.
#workflow: Opens the ROOT file and accesses the mu3e tree - array of all frames and the pixelIDs
# Extracts hit data (hit_pixelid) for the specified frame.
# Loops through each hit, decodes it using the Hit class, and appends its details to a list frame_hits.



    hits_data_frame = pd.DataFrame(frame_hits)
    #print(hits_data_frame)  # Display the DataFrame for verification

    return hits_data_frame  # Return the DataFrame for further use
    #print(frame_hits)
    
# Converts the list to a Pandas DataFrame for easy handling. (Essentially making hit position intuitively human readable)
#output: DataFrame containing the decoded hit information.



if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("You need to specify an input file(s)")
        sys.exit(-1)

    for argument in sys.argv[1:]:
        print("File:", argument)
        printHitsInFirstFrame(argument)
        #What's going on here? *************

#######################################################################################################

# Inputting a file and testing the output
file_path = "/app/Simulation Data/v5.3/signal1_96_32652_execution_1_run_num_789062_sort.root"
print("Test with the file:", file_path) 
print()

#Find number of frames have:
signal_file = uproot.open(file_path)
frames = signal_file['mu3e'].arrays()
total_frames = len(frames)
print('Total number of frames in file:',total_frames)
print()
#Opens the ROOT file and calculates the total number of frames

# Create directory for file using
directory = "/app/ProcessedData/signal1_96_32652_execution_1_run_num_789062_sort.root" 
# Directory : NOTE currently this needs to be changed each time create csv files, for new sort file, can try automate

if not os.path.exists(directory):
    os.makedirs(directory)
#Creates a directory to store the results.

file_name = "hits_data_(signal1_96_32652_execution_1_run_num_789062_sort.root).csv"
if os.path.exists(file_name): # Deletes old version of file if present
    os.remove(file_name)
#Deletes any pre-existing file with the same name.


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

    #What are these if else functions doing here? ****************

    #print(frame_hits_data)


