# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:12:35 2024

@author: nicky
"""

import uproot 
import matplotlib.pyplot as plt
import numpy as np

file = uproot.open("/root/Mu3eProject/RawData/signal1_1_1944629_execution_1_run_num_836827_vertex.root")

vertices = file["vertex"].arrays()


######## Playing around with uproot calling commands #######
print(vertices.fields)
#print()
#print(vertices.true_px1.show)

# number_list = [] # List to count through and ammend the number of mass events

# for i in range(3800, 4500):
#     print(len(vertices[i].mmass))

#############################################################################


## Code to collect all the mass events into a single array ################

overall_mass_array = np.empty(0) # Initializes an empty array to populate with the masses

for i in range(len(vertices)):           # Iterate through frames and count mass events, and add
   frame_mass_array = (vertices[i].mmass)
    # print(number_per_frame)
   overall_mass_array = np.append(overall_mass_array,frame_mass_array) # Append the mass array with the new frame values

#print(overall_mass_array) # Overall mass array for all of the events recorded in the data
#print(vertices[1].mmass)
#print(file["vertex"][3].mmass)

#np.sort(overall_mass_array)
print(overall_mass_array)

ordered_mass_array = np.sort(overall_mass_array)

print(ordered_mass_array)




bin_width = 0.2
bin_edges = np.arange(0, 170, bin_width) # Realised I don't actually know what this is doing

# 
plt.hist(overall_mass_array, bins=bin_edges, edgecolor = 'black')

plt.xlabel('mass')
plt.ylabel('Frequency')
plt.title('Mass frequency')


counts, bin_edges, patches = plt.hist(overall_mass_array, bins=np.arange(104,106,bin_width), edgecolor = 'blue') # another bit of histogram plot over small area looking at 
plt.show()


print('hello world')


