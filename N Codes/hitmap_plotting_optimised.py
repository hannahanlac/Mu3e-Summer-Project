import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import hist 

####### Global variables ############
pixel_size_y = 250 
pixel_size_x = 256 

layer_properties = {
    1:{"ladders": 8, "chips": 6},
    2:{"ladders": 10, "chips": 6},
    3:{"ladders": 24, "chips": 17},
    4:{"ladders": 28, "chips": 18},
    }

##############################################################
def frameHitDataPrep(frame_hits_data, layer, frame_number):
    """Function for plotting hitmaps for each frame. 
    Input: Panda of hitmap data for given frame
    Output: Hitmaps for each layer""" #NOTE: Only currently works for layer 1 - 4 not for upstream/downstream stations


    layer_n_hits = frame_hits_data[frame_hits_data['layer'] == layer] #Extract the hits information for the given layer
    ladder_max = layer_properties[layer]["ladders"]
    chip_max = layer_properties[layer]["chips"]

    # Dimensions
    pixel_x_max = chip_max * pixel_size_x
    pixel_y_max = ladder_max * pixel_size_y 

    hit_x_positions_absolute = layer_n_hits['pixel_x'] + (layer_n_hits['chip']-1)*pixel_size_x
    hit_y_positions_absolute = layer_n_hits['pixel_y'] + (layer_n_hits['ladder']-1)*pixel_size_y

    return hit_x_positions_absolute, hit_y_positions_absolute, pixel_x_max, pixel_y_max, chip_max, ladder_max

def HitmapPlotting (layer, hit_x_positions_absolute, hit_y_positions_absolute, pixel_x_max, pixel_y_max, chip_max, ladder_max):
    # heatmap bin sizes - again need to change a bit as gpt helped
    bin_size_x = pixel_size_x / 10
    bin_size_y = pixel_size_y / 10
    x_bins = int(pixel_x_max / bin_size_x)
    y_bins = int(pixel_y_max / bin_size_y)

    #heatmap 
    heatmap, xedges, yedges = np.histogram2d(
    hit_x_positions_absolute, hit_y_positions_absolute, bins=[x_bins, y_bins]
    )
     # Define the figure and plot
    # plt.figure(figsize=(10, 8))
    # #plt.scatter(hit_x_positions_absolute, hit_y_positions_absolute, color='red', s=20, label='Hit') # Scatter from doing 1 frame at time
    # plt.imshow(heatmap.T, cmap='hot', interpolation='nearest')

    # Working code for heatmap: Need to update as some GPT helped
    plt.figure(figsize=(10, 8))
    plt.imshow(
        heatmap.T, 
        cmap='hot', 
        interpolation='nearest', 
        origin='lower',  # To align with physical positions
        extent=[0, pixel_x_max, 0, pixel_y_max])  # Match the physical dimensions


    # Set the axis limits
    plt.xlim(0, pixel_x_max)
    plt.ylim(0, pixel_y_max)

    # Add axis labels and title
    plt.xlabel("Chip", fontsize=14)
    plt.ylabel("Ladder", fontsize=14)
    plt.title(f"All frames Layer {layer} hits heatmap", fontsize=16)

    #Add custom ticks and labels for ladder (y-axis) and chip (x-axis) This needs editing
    plt.xticks(
        ticks=[i * pixel_size_x for i in range(chip_max)],
        labels=[str(i + 1) for i in range(chip_max)],
        fontsize=12
    )
    plt.yticks(
        ticks=[i * pixel_size_y  for i in range(ladder_max)],
       labels=[str(i + 1) for i in range(ladder_max)],
       fontsize=12
    )

    # Add a grid for better visualization
    #plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show the plot
    plt.legend()
    plt.show()
 
def layerHistPlottingFast (frame_hits_data):
   """Better function for extracting hit info and plotting"""
   hits_per_layer = frame_hits_data['layer'].value_counts().sort_index()
   print(hits_per_layer)

    # Create a bar plot
   plt.figure(figsize=(8, 6))
   plt.bar(hits_per_layer.index, hits_per_layer.values, color='blue', edgecolor= 'black')
    
       # Add labels and title
   plt.xlabel('Layer', fontsize=14)
   plt.ylabel('Number of Hits', fontsize=14)
   plt.title('Hit Counts Per Layer', fontsize=16)

  # Annotate each bar with the hit count
   for index, value in zip(hits_per_layer.index, hits_per_layer.values):
      plt.text(index, value + 5, str(value), ha='center', fontsize=12)

  # Customize tick labels
      plt.xticks(hits_per_layer.index, labels=[f"Layer {int(layer)}" for layer in hits_per_layer.index], fontsize=12)

  # Show the plot
   plt.tight_layout()
   plt.show()


# Specify range of frames plots wanted:
#frame_numbers = list(range(1,3))

######### For doing one frame at a time: #######
# for frame_number in frame_numbers:
#     file_name = "/root/Mu3eProject/WorkingVersion/Mu3eProject/Frame_hits_csvs_signal1_1_1944629/hits_data_frame{}.csv".format(frame_number) # Open csv for frame number as panda. NOTE: This will only work if you have already created the hit data csv for the specific frame.
#     frame_hits_data = pd.read_csv(file_name) 
#     print('Frame number:', frame_number)
#     print(frame_hits_data)
#     frameHitPlotting(frame_hits_data, 1, frame_number)
#     frameHitPlotting(frame_hits_data, 2, frame_number)
#     frameHitPlotting(frame_hits_data, 3, frame_number) # Note currently not doing up/down stream recurl stations
#     frameHitPlotting(frame_hits_data, 4, frame_number)

# For doing for a whole file: ### 



file_name = "/root/Mu3eProject/WorkingVersion/Mu3eProject/DataFilesV5.3/signal1_99_32652/hits_data_signal1_99_32652.csv"
frame_hits_data = pd.read_csv(file_name)

frame_number = 1 # This is superfluous but will use to get working (as now going over whole file)

for layer in range(1, 5):
     hit_x, hit_y, x_max, y_max, chip_max, ladder_max = frameHitDataPrep(frame_hits_data, layer, frame_number)
     HitmapPlotting(layer, hit_x, hit_y, x_max, y_max, chip_max, ladder_max)


#layerHistPlottingSlow(frame_hits_data)
layerHistPlottingFast(frame_hits_data)


