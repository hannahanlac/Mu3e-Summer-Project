# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:05:12 2024

@author: m4joh
"""

import uproot
import matplotlib.pyplot as plt
import numpy as np

file = uproot.open("C:/Users/m4joh/OneDrive/University/Mu3e/Raw Simulation Data/signal1_0_1944629_execution_1_run_num_407942_vertex.root")
vertices = file["vertex"].arrays()

for i in range(100):
    Frame_vertex = vertices[i].mmass
    print(Frame_vertex)  
#Prints masses of different vertices in the first e.g. 100 frames
#E.g. finds 2 vertices and prints the masses

print(len(vertices))
#"vertices[0]" is first frame, "vertices[1]" the second etc
#'vertices' = an array, one entry for each frame.
#code counting number of array entries i.e. frames in array/sample
#9921 frames in sample

#Next
#Count number of events/vertices/mmasses (same thing) in total sample
#Plot number of events/vertices/mmasses per frame

Frames = np.arange(0, len(vertices))
print(Frames)

number_list = []
#number of vertices in each frame - do in list as can't create an empty array, must populate with a zero
#so just use empty list and convert to array later (Can create empty array - see later)

for i in range(len(vertices)):
    number_per_frame = len(vertices[i].mmass)
    #print(number_per_frame)
    number_list.append(number_per_frame)
#for every value in list (up to number of frames in sample)
#count the number of masses(vertices) and add that number to the list - corresponding to the number frame 
# just scanned
    
number_array = np.array(number_list)
#convert list to array of number of events/mmass reconstructions per frame

print(np.max(number_array))
#plots the maximum number of vertices found in a frame
#ANS - one frame counted as having 103 events in the frame
# However, this = sus as have seen that when try do this for a large number of frames (e.g. all 9921),
# code breaks and makes mistakes accounting for how many events in what frames

print(number_array)
#plots the number array - number of vertices per frame
#can match with print(Frames) to see number of vertices next to number of frames



plt.bar(Frames, number_array)
plt.ylim(0, 28)
plt.xlim(0, 9920)
plt.xlabel('Frame')
plt.ylabel('Number of Vertices')
plt.title('Number of Events per Frame')

plt.show()

#If plot for a large number of frames, code very inefficient and breaks - 
#doesn't count the number of events per frame properly (ie. doesn't match arrays)
#should be way to do it efficiently e.g. by binning code but need move on now
#in future, if need to look at this, select a smaller subset of frames and just plot for those -
#code is then accurate (check a few values next to arrays anyway just to be safe)



#Code is very inefficient and excessive data points is breaking code giving
#issues plot when try programme all of them

#Find way to bin more efficiently so plot accurately
#Find way to count events in way can keep track of them
#Plot chi2, target distance and mmass against events
#Cut out events based on certain parameters with above metrics
#e.g. big mass dif from @ rest muon, large(?) target dist?