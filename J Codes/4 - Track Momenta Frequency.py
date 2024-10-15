# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:22:49 2024

@author: m4joh
"""

import uproot
import matplotlib.pyplot as plt
import numpy as np

file = uproot.open("C:/Users/m4joh/OneDrive/University/Mu3e/Raw Simulation Data/signal1_0_1944629_execution_1_run_num_407942_vertex.root")
vertices = file["vertex"].arrays()

print(dir(vertices[0]))


#Track 1 Momentum Frequency

# for i in range(len(vertices)):
#     eventtrack1p = vertices[i].p1
#     print(eventtrack1p) 

track1p_array = np.empty(0)

for i in range(len(vertices)):
    eventtrack1p = vertices[i].p1
    track1p_array = np.append(track1p_array, eventtrack1p)
    
print(len(track1p_array))

#print(track1p_array)

track1p_array = abs(track1p_array)

plt.hist(track1p_array, bins=100, edgecolor = 'black')
plt.xlabel('Track 1 Momentum')
plt.ylabel('Frequency')
plt.title('Track 1 Momentum Frequency')

plt.show()


#Track 2 Momentum Frequency

# for i in range(len(vertices)):
#     eventtrack2p = vertices[i].p2
#     print(eventtrack2p) 

track2p_array = np.empty(0)

for i in range(len(vertices)):
    eventtrack2p = vertices[i].p2
    track2p_array = np.append(track2p_array, eventtrack2p)
    
print(len(track2p_array))

#print(track2p_array)

track2p_array = abs(track2p_array)

plt.hist(track2p_array, bins=100, edgecolor = 'black')
plt.xlabel('Track 2 Momentum')
plt.ylabel('Frequency')
plt.title('Track 2 Momentum Frequency')

plt.show()


#Track 3 Momentum Frequency

# for i in range(len(vertices)):
#     eventtrack3p = vertices[i].p3
#     print(eventtrack3p) 

track3p_array = np.empty(0)

for i in range(len(vertices)):
    eventtrack3p = vertices[i].p3
    track3p_array = np.append(track3p_array, eventtrack3p)
    
print(len(track3p_array))

#print(track3p_array)

track3p_array = abs(track3p_array)

plt.hist(track3p_array, bins=100, edgecolor = 'black')
plt.xlabel('Track 3 Momentum')
plt.ylabel('Frequency')
plt.title('Track 3 Momentum Frequency')

plt.show()