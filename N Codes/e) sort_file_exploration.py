import uproot
import numpy
import matplotlib.pyplot as plt
import awkward as ak

file = uproot.open(r"/app/Simulation Data/v5.3/signal1_95_32652_execution_1_run_num_67021_sort.root")

keys = file.keys() # Accessing the file keys to see what info there is
#lists datasets and structures stored in root file

###print("Keys in the ROOT file:", keys)
#very long list

hits1 = file["mu3e;1"].arrays() #This is my current best guess at what the hit information is 
#mu3e = particulaar data set in root file e.g., guessing this possibly = detector hits
#;1 = dataset version (1 = first/latest version)
#name it hits1 as guess it = hit info and it = version 1

###print(hits1) # Printing the first part of hits1: 1st frame hit info?
print(hits1.fields) # Finding the fields in hit info
#Fields correspond to individual variables recorded for each entry in the dataset.
#e.g., for every hit, info may be recorded on e.g., timing (fibres), trajectory, etc
#Here simply printing all these different fields of data can extract from each hit


###print(len(hits1)) # Verifying length hits1 - Does appear to be correct number of frames?
#prints 9921 - number of frames previously counted to be in these files

print()

#Trying to access 'hits pixelid' - Although still unsure what this is.
print(hits1[0])
#prints all hit(?) info in frame 1? Guessing entry no. correllates to frames as no. matches no. frames in file
#prints a lot of info, indicates that each entry is a dictionary containing arrays for each field.
#i.e., each entry (representing each frame) contains a dictionary of every fields and arrays of these field values for each hit

print()


hit_pixelid = hits1[0]["hit_pixelid"]
#Extracts the value of the field "hit_pixelid" from the first (0) entry in the dataset
#prints hit pixelid dictionary array only

print(hit_pixelid)

#################################################################################

# Testing with variables obtained from extracting_hits.py to ensure extracting correct information
frame_number = 0
#just specifying frame looking at for later reference

mu3eTree = file['mu3e'].arrays()
#Loads the entire dataset (tree) associated with the mu3e key into memory - holds data from mu3e tree (dataset) as a dictionary with keys
#.arrays() method converts ROOT data into Python-friendly arrays (e.g., NumPy or Awkward Arrays).

mu3eFrame = ak.Array([mu3eTree[frame_number]])
#Isolates the data for the specified frame (frame_number = 0) and wraps it in an Awkward Array.
#mu3eTree[frame_number] extracts the data for the frame.
#ak.Array([...]) converts the data into an Awkward Array, which is well-suited for handling hierarchical or jagged datasets
#Awkward Arrays can represent complex, nested data structures.

total_frames = len(mu3eTree)
#total number of frames (or entries) in the mu3eTree dataset.

print("The 0 frame hit data:" , mu3eFrame)
#The output data is shown as a nested structure, consistent with hierarchical or event-based data.

print("The total number of frames is:", total_frames)