# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:30:24 2024

@author: m4joh
"""

import uproot
import matplotlib.pyplot as plt
import numpy as np

file = uproot.open("C:/Users/m4joh/OneDrive/University/Mu3e/Raw Simulation Data/signal1_0_1944629_execution_1_run_num_407942_vertex.root")
vertices = file["vertex"].arrays()

number_list = []
#number of vertices in each frame - do in list as can't create an empty array, must populate with a zero
#so just use empty list and convert to array later

for i in range(len(vertices)):
    number_per_frame = len(vertices[i].mmass)
    #print(number_per_frame)
    number_list.append(number_per_frame)
#for every value in list (up to number of frames in sample)
#count the number of masses(vertices) and add that number to the list - corresponding to the number frame just scanned
    
number_array = np.array(number_list)
#converting 

print(np.sum(number_array))
#gives number of events in sample

event_number = np.arange(1, 20618)
#creating an array to track every event in sample

print(event_number)


vertex_mass_array = np.empty(0)
#creating an empty array

for i in range(len(vertices)):
    vertex_mass = vertices[i].mmass
    vertex_mass_array = np.append(vertex_mass_array, vertex_mass)
    #print(vertex_mass)  
#Prints masses of different vertices in the first e.g. 100 frames
#E.g. finds 2 vertices and prints the masses

print(len(vertex_mass_array))
#number of masses in array
print(vertex_mass_array)
#masses array


#Want to create 2D array dictionary where each mass is matched to the frame
#it comes from, that way can find frame interesting events occurred in


plt.hist(vertex_mass_array, bins=100, edgecolor = 'black')
plt.xlabel(r'Mass Value [MeV/c$^2$]')
plt.ylabel('Frequency')
plt.title('Mass Frequency')

plt.show()





