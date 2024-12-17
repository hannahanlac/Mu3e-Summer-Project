import uproot
import numpy
import matplotlib.pyplot as plt
import awkward as ak

file = uproot.open("/root/Mu3eProject/RawData/v5.3/signal1_95_32652_execution_1_run_num_67021_sort.root")

keys = file.keys() # Accessing the file keys to see what info there is
#print("Keys in the ROOT file:", keys)

hits1 = file["mu3e;1"].arrays() #This is my current best guess at what the hit information is 

#print(hits1) # Printing the first part of hits1: 1st frame hit info?
print(hits1.fields) # Finding the fields in hit info


# print(len(hits1)) # Verifying length hits1 - Does appear to be correct number of frames?


#Trying to access 'hits pixelid' - Although still unsure what this is.
print("The 1 frame hit data is:",hits1[1])
print()
hit_pixelid = hits1[1]["hit_pixelid"]
print()
print(hit_pixelid)

print()
frame_number = 0
mu3eTree = file['mu3e'].arrays()
mu3eFrame = ak.Array([mu3eTree[frame_number]])

total_frames = len(mu3eTree)

print("The 0 frame is:" , mu3eFrame)
print("The total number of frames is:", total_frames)