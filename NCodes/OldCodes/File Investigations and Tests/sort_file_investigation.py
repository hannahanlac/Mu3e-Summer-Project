import uproot
import numpy
import matplotlib.pyplot as plt
import awkward as ak

#### Code I am using to investigate sort files and how to extract the correct information from them. This is mainly just testing. ####

file = uproot.open("/root/Mu3eProject/RawData/v5.3/signal1_99_32652_execution_1_run_num_561343_sort.root")


hits1 = file["mu3e"].arrays() #This is my current best guess at what the hit information is 
print(hits1.fields) # Finding the fields in hit info: Does appear to be for hit info

#print(hits1["alignment"])

# print(hits1) # Printing the first part of hits1: All of the hit information
# print()


# #Trying to access 'hits pixelid' - Although still unsure what this is.
# print("The 1 frame hit data is:",hits1[0])
# hit_pixelid = hits1[0]["hit_pixelid"]
# print("Pixel IDs:", hit_pixelid)
# print()

# #Trying to access 'hits_timestamp' - Finding hit timestamps
# print("The 1 frame hit data is:",hits1[1])
# hit_pixeltimestamp = hits1[1]["hit_timestamp"]
# print("Pixel Timestamps:", hit_pixeltimestamp)
# print()



# Testing with variables from extracting_hits.py to ensure extracting correct information
frame_number = 0
mu3eTree = file['mu3e'].arrays()
mu3eFrame = ak.Array([mu3eTree[frame_number]])
mu3eFrameTime = mu3eFrame["hit_timestamp"]
mu3eFrameMCIndex = mu3eFrame["hit_mc_i"]
mu3eFrameMCNumber = mu3eFrame["hit_mc_n"]
mu3eFramePixelIds = mu3eFrame["hit_pixelid"]

# total_frames = len(mu3eTree)

# print("The 0 frame is:" , mu3eFrame)
# print("The 0 frame time info is:", mu3eFrameTime)
print("The 0 frame mc_indexes are:", mu3eFrameMCIndex)
print("The 0 frame mc_numbers are:", mu3eFrameMCNumber)
print("The number of hits in the frame is:", len(mu3eFrame["hit_pixelid"]))


# print("The total number of frames is:", total_frames)
