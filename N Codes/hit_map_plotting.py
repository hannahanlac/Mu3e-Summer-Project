import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np


# Plotting the hits data:

def frameHitPlotting (frame_hits_data, layer, frame_number):
    """Function for plotting hitmaps for each frame. 
    Input: Panda of hitmap data for given frame
    Output: Hitmaps for each layer""" #NOTE: Only currently works for layer 1
    n = int(layer)
    layer_n_hits = frame_hits_data[frame_hits_data['layer'] == n]

    if layer == 1:
      ladder_max = 8 #No.Ladders from TDR layer 1
      chip_max = 6 #No. Pixel chips per layer layer 1 TDR
    
    elif layer == 2:
      ladder_max = 10 #No.Ladders from TDR layer 2
      chip_max = 6 #No. Pixel chips per layer layer 2 TDR

    elif layer ==3:
       ladder_max = 24
       chip_max = 17
                            # NOTE: For 3 and 4 need to check the ladder numbers are correct for JUST the central barrel.
    elif layer == 4:
       ladder_max = 28
       chip_max = 18

    pixel_size = 250 # This needs to be checked in meeting 
    
    # Really need to know the x and y max of these pixels! Think I can guess at 250 from the hitmap data Mark has given me?? 
    pixel_x_max = chip_max* pixel_size
    pixel_y_max = ladder_max * pixel_size 
    hit_x_positions_absolute = layer_n_hits['pixel_x'] + (layer_n_hits['chip']-1)*pixel_size
    hit_y_positions_absolute = layer_n_hits['pixel_y'] + (layer_n_hits['ladder']-1)*pixel_size
    
    print(hit_x_positions_absolute)
    print(hit_y_positions_absolute)
    

     # Define the figure and plot
    plt.figure(figsize=(10, 8))
    plt.scatter(hit_x_positions_absolute, hit_y_positions_absolute, color='red', s=20, label='Hit')

    # Set the axis limits
    plt.xlim(0, pixel_x_max)
    plt.ylim(0, pixel_y_max)

    # Add axis labels and title
    plt.xlabel("Chip", fontsize=14)
    plt.ylabel("Ladder", fontsize=14)
    plt.title(f"Frame {frame_number} Layer {n} Hit Positions", fontsize=16)

    #Add custom ticks and labels for ladder (y-axis) and chip (x-axis)
    plt.xticks(
        ticks=[i * pixel_size for i in range(chip_max)],
        labels=[str(i + 1) for i in range(chip_max)],
        fontsize=12
    )
    plt.yticks(
        ticks=[i * pixel_size  for i in range(ladder_max)],
       labels=[str(i + 1) for i in range(ladder_max)],
       fontsize=12
    )

    # Add a grid for better visualization
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show the plot
    plt.legend()
    plt.show()
 
# Specify range of frames plots wanted:
frame_numbers = list(range(1,10))


for frame_number in frame_numbers:
    file_name = "/root/Mu3eProject/WorkingVersion/Mu3eProject/Frame_hits_csvs_signal1_1_1944629/hits_data_frame{}.csv".format(frame_number) # Open csv for frame number as panda. NOTE: This will only work if you have already created the hit data csv for the specific frame.
    frame_hits_data = pd.read_csv(file_name) 
    print('Frame number:', frame_number)
    print(frame_hits_data)
    frameHitPlotting(frame_hits_data, 1, frame_number)
    frameHitPlotting(frame_hits_data, 2, frame_number)
    frameHitPlotting(frame_hits_data, 3, frame_number) # Note currently not doing up/down stream recurl stations
    frameHitPlotting(frame_hits_data, 4, frame_number)


