import uproot
import numpy
import matplotlib.pyplot as plt

file = uproot.open("/root/Mu3eProject/RawData/HitData/signal1_1_1944629_execution_1_run_num_836827_sort.root")

keys = file.keys() # Accessing the file keys to see what info there is
#print("Keys in the ROOT file:", keys)

hits1 = file["mu3e;1"].arrays() #This is my current best guess at what the hit information is 

#print(hits1) # Printing the first part of hits1: 1st frame hit info?
print(hits1.fields) # Finding the fields in hit info


# print(len(hits1)) # Verifying length hits1 - Does appear to be correct number of frames?


#Trying to access 'hits pixelid' - Although still unsure what this is.
print(hits1[1])
print()
hit_pixelid = hits1[1]["hit_pixelid"]
print()
print(hit_pixelid)


