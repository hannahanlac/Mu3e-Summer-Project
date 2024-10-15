# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:18:43 2024

@author: m4joh
"""

import uproot
import matplotlib.pyplot as plt
import numpy as np

file = uproot.open("C:/Users/m4joh/OneDrive/University/Mu3e/signal1_0_1944629_execution_1_run_num_407942_vertex.root")
vertices = file["vertex"].arrays()

event_totalp_array = np.empty(0)

for i in range(len(vertices)):
    event_p1 = abs(vertices[i].p1)
    event_p2 = abs(vertices[i].p2)
    event_p3 = abs(vertices[i].p3)
    event_totalp = event_p1 + event_p2 + event_p3
    event_totalp_array = np.append(event_totalp_array, event_totalp)
    
#print(event_totalp_array)
print(len(event_totalp_array))



event_total_truep_array = np.empty(0)

for i in range(len(vertices)):
    event_truep1 = vertices[i].true_p1
    event_truep2 = vertices[i].true_p2
    
    truepx3 = abs(vertices[i].true_px3)
    truepy3 = abs(vertices[i].true_py3)
    truepz3 = abs(vertices[i].true_pz3)
    
    event_truep3 = np.sqrt(truepx3**2 + truepy3**2 + truepz3**2) 
    event_total_truep = event_truep1 + event_truep2 + event_truep3
    event_total_truep_array = np.append(event_total_truep_array, event_total_truep)
    
#print(event_total_truep_array)
print(len(event_total_truep_array))






# Define the position of the bars on the x-axis
#x = np.arange(len(vertices))
x = np.arange(20) #Only plot 1st 100 events

# Width of the bars
width = 0.4

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the first bar chart (background) with transparency
ax.bar(x, event_total_truep_array[:20], width, label='True Momentum', alpha=0.5, color='red', edgecolor='black')

# Plot the second bar chart (foreground) opaque
ax.bar(x, event_totalp_array[:20], width, label='Reconstructed Momentum', alpha=0.5, color='grey', edgecolor='black')

# Add labels and title
ax.set_ylabel('Momentum')
ax.set_xlabel('Event Number')
ax.set_title('Reconstructed vs True Total Momentum')

# Add a legend
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()

#Note: Reason we have to plot in this way, is because we wnat the bar charts
#to overlap. What I've essentially done here is start by establishing an x axis
# Then plotting two different bar charts onto the same axis with transparency.
#Could probably do better e.g. with Hist package where the charts are
#intrinsically tied to one another, not just happen to be on top of each other
# - seems a bit fragile