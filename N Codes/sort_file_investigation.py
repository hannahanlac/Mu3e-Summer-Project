import uproot
import numpy
import matplotlib.pyplot as plt
import awkward as ak

#### Code I am using to investigate sort files and how to extract the correct information from them. This is mainly just testing. ####

file = uproot.open("/root/Mu3eProject/RawData/v5.3/signal1_95_32652_execution_1_run_num_67021_sort.root")

keys = file.keys() # Accessing the file keys to see what info there is
# print("Keys in the ROOT file:", keys)
# print()

hits1 = file["mu3e"].arrays() #This is my current best guess at what the hit information is 
print(hits1.fields) # Finding the fields in hit info: Does appear to be for hit info

print(hits1) # Printing the first part of hits1: All of the hit information
print()


#Trying to access 'hits pixelid' - Although still unsure what this is.
print("The 1 frame hit data is:",hits1[1])
print()
hit_pixelid = hits1[1]["hit_pixelid"]
print()
print("Pixel IDs:", hit_pixelid)


# Testing with variables from extracting_hits.py to ensure extracting correct information
frame_number = 0
mu3eTree = file['mu3e'].arrays()
mu3eFrame = ak.Array([mu3eTree[frame_number]])

total_frames = len(mu3eTree)

print("The 0 frame is:" , mu3eFrame)
print("The total number of frames is:", total_frames)