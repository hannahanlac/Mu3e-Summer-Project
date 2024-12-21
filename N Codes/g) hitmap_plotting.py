import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import hist 


def frameHitPlotting (frame_hits_data, layer, frame_number, station):
    """Function for plotting hitmaps for each frame. 
    Input: Panda of hitmap data for given frame
    Output: Hitmaps for each layer""" 

    '''Generates hitmaps for each frame, for each layer, for each station
    Parameters: frame_hits_data: Frame containing all hit data.
    layer: The detector layer being analyzed.
    frame_number: The specific frame plotting for.
    station: Specifies the detector station (e.g., central, upstream, downstream)'''

    n = int(layer)
    layer_n_hits = frame_hits_data[frame_hits_data['layer'] == n]
    # Filters the input data to include only hits corresponding to the specified layer.

    if layer == 1:
      ladder_max = 8 #No.Ladders from TDR layer 1
      chip_max = 6 #No. Pixel chips per ladder layer 1 TDR
    
    elif layer == 2:
      ladder_max = 10 #No.Ladders from TDR layer 2
      chip_max = 6 #No. Pixel chips per ladder layer 2 TDR

    elif layer ==3:
       ladder_max = 24
       chip_max = 17
                            # NOTE: For 3 and 4 need to check the ladder numbers are correct for JUST the central barrel.
    elif layer == 4:
       ladder_max = 28
       chip_max = 18

#Configures the number of ladders and chips based on the selected layer.

    total_chips_central = chip_max
    total_chips = 3 * chip_max  # Upstream + Central + Downstream
    #Again just specifying useful information about detector build


    # Specifying the station used
    if station ==0:
       layer_n_hits = layer_n_hits[layer_n_hits['station'] == 0] # Just use data central barrel
       station_name = "central barrel"

    elif station ==1:
        layer_n_hits = layer_n_hits[layer_n_hits['station'] == 1] # upstream data only
        station_name = "upstream recurl station"

    elif station ==2:
        layer_n_hits = layer_n_hits[layer_n_hits['station'] == 2] # Downstream data only
        station_name = "downstream recurl station"

    elif station == "all":  
      station_name = "recurl stations and central barrel"
      chip_max = total_chips
      chip_offset = layer_n_hits['station'].map({
          1: 0,  # Upstream starts at 0
          0: total_chips_central,  # Central starts after upstream
          2: 2 * total_chips_central  # Downstream starts after central
        })
#Again, configuring data further by specifying what station want it for
#What does the chip_offset function do? *************
    

    pixel_size_y = 250 # Checked in meeting and this correct
    pixel_size_x = 256 
    #pixel dimensions

    # Define absolute hit positions
    if station in [0, 1, 2]:  # Single station case, no offset
        hit_x_positions_absolute = layer_n_hits['pixel_x'] + (layer_n_hits['chip'] - 1) * pixel_size_x

    else:  # Combined stations case, apply offset
        hit_x_positions_absolute = layer_n_hits['pixel_x'] + ((layer_n_hits['chip'] - 1) + chip_offset) * pixel_size_x

    hit_y_positions_absolute = layer_n_hits['pixel_y'] + (layer_n_hits['ladder'] - 1) * pixel_size_y
#Computes absolute x and y positions of hits based on chip and ladder.
#Why x and y treated differently depending on if single station or combined case? ************
    
    ### print(hit_x_positions_absolute)
    ### print(hit_y_positions_absolute)
    
    # heatmap bin sizes - again need to change a bit as gpt helped
    bin_size_x = pixel_size_x /2
    bin_size_y = pixel_size_y /2
    x_bins = int(chip_max * pixel_size_x / bin_size_x)
    y_bins = int(ladder_max * pixel_size_y / bin_size_y)
    #Defines heatmap resolution by determining bin sizes in x and y directions
    #Where formula come from? *****************************


    #heatmap 
    heatmap, xedges, yedges = np.histogram2d(
    hit_x_positions_absolute, hit_y_positions_absolute, bins=[x_bins, y_bins]
    )
# Creates a 2D histogram to represent hit density.
#How defining the histogram actually work here? ****************


    # Define the figure and plot
    ### plt.figure(figsize=(10, 8))
    ### plt.scatter(hit_x_positions_absolute, hit_y_positions_absolute, color='red', s=20, label='Hit') # Scatter from doing 1 frame at time
    ### plt.imshow(heatmap.T, cmap='hot', interpolation='nearest')

    # Working code for heatmap: Need to update as some GPT helped
    plt.figure(figsize=(14, 8))
    heatmap_plot =plt.imshow(
        heatmap.T, 
        cmap='hot', 
        interpolation='nearest', 
        origin='lower',  # To align with physical positions
        extent=[0, chip_max * pixel_size_x, 0, ladder_max * pixel_size_y])  # Match the physical dimensions
#Visualizes the heatmap with a specific color map, alignment, and scaling.
#Don't really get how this function works though ******************


    # Set the axis limits
    plt.xlim(0, chip_max * pixel_size_x)
    plt.ylim(0, ladder_max * pixel_size_y)
    #i.e. size of axes

    # Add axis labels and title
    plt.xlabel("Chip", fontsize=14)
    plt.ylabel("Ladder", fontsize=14)
    plt.title(f"All frames Layer {n} {station_name} heatmap", fontsize=16)

    #Add custom ticks and labels for ladder (y-axis) and chip (x-axis) This needs editing
    plt.xticks(
        ticks=[i * pixel_size_x for i in range(chip_max)],
        labels=[str(i + 1) for i in range(chip_max)],
        fontsize=10
    )
    plt.yticks(
        ticks=[i * pixel_size_y  for i in range(ladder_max)],
       labels=[str(i + 1) for i in range(ladder_max)],
       fontsize=12
    )
    #Structuring mapping interms of axis ticks/labels

    # Add a grid for better visualization
    ### plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show the plot
    plt.legend()
    plt.show()
 
# Specify range of frames plots wanted for:
###frame_numbers = list(range(1,3))

######### For doing one frame at a time: #######
# for frame_number in frame_numbers:
#     file_name = "/root/Mu3eProject/WorkingVersion/Mu3eProject/Frame_hits_csvs_signal1_1_1944629/hits_data_frame{}.csv".format(frame_number) 
# Alter file name to your csv? ***************
# Open csv for frame number as panda. 
# NOTE: This will only work if you have already created the hit data csv for the specific frame.
#     frame_hits_data = pd.read_csv(file_name) 
#     print('Frame number:', frame_number)
#     print(frame_hits_data)
#Now call function just been defining and use it, specifying parameters to use it on
#     frameHitPlotting(frame_hits_data, 1, frame_number)
#     frameHitPlotting(frame_hits_data, 2, frame_number)
#     frameHitPlotting(frame_hits_data, 3, frame_number) # Note currently not doing up/down stream recurl stations
#     frameHitPlotting(frame_hits_data, 4, frame_number)




# For doing for a whole file: (Much better)

def layerHistPlottingSlow (frame_hits_data):
  """Function that plots histogram of layer hits"""
  #Named slow as very inefficient, fast is much more efficient
  #n = int(layer)
  layer_1_hits, layer_2_hits, layer_3_hits, layer_4_hits, layer_3_up, layer_3_down, layer_4_up, layer_4_down = (
     frame_hits_data[frame_hits_data['layer'] == 1], 
     frame_hits_data[frame_hits_data['layer'] == 2],
     frame_hits_data[frame_hits_data['layer'] == 3], 
     frame_hits_data[frame_hits_data['layer'] == 4]
  )
  #What is this doing? *****************

  length_1 = len(layer_1_hits)
  length_2 = len(layer_2_hits)
  length_3 = len(layer_3_hits)
  length_4 = len(layer_4_hits)
  #counting number of hits in each layer (for all frames, as specified above)

  print(layer_1_hits, layer_2_hits, layer_3_hits, layer_4_hits)
  print(length_1, length_2, length_3, length_4)

 # Data for plotting
  layers = [1, 2, 3, 4]
  hit_counts = [length_1, length_2, length_3, length_4]

  # Create the bar plot
  plt.figure(figsize=(8, 6))
  plt.bar(layers, hit_counts, color='skyblue', edgecolor='black')

  # Add labels and title
  plt.xlabel('Layer', fontsize=14)
  plt.ylabel('Number of Hits', fontsize=14)
  plt.title('Hit Counts Per Layer', fontsize=16)

    # Annotate bars with the hit count values
  for i, count in enumerate(hit_counts):
      plt.text(layers[i], count + 5, str(count), ha='center', fontsize=12)
#What is this? ************

    # Set x-axis ticks to match layers
  plt.xticks(layers, labels=[f"Layer {layer}" for layer in layers], fontsize=12)

  # Show the plot
  plt.tight_layout()
  plt.show()

def layerHistPlottingFast (frame_hits_data):
   """Better function for extracting hit info and plotting"""
   hits_per_layer = frame_hits_data['layer'].value_counts().sort_index()
   #What is the sorting being done here? *************
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
      #What is this? ************
      plt.text(index, value + 5, str(value), ha='center', fontsize=12)

  # Customize tick labels
      plt.xticks(hits_per_layer.index, labels=[f"Layer {int(layer)}" for layer in hits_per_layer.index], fontsize=12)

  # Show the plot
   plt.tight_layout()
   plt.show()


file_name = "/app/ProcessedData/signal1_96_32652/csv/hits_data_signal1_96_32652.csv"
frame_hits_data = pd.read_csv(file_name)

frame_number = 1 # This is superfluous but will use to get working (as now going over whole file)
# frameHitPlotting(frame_hits_data, 1, frame_number, 0)
# frameHitPlotting(frame_hits_data, 2, frame_number, 0)
frameHitPlotting(frame_hits_data, 3, frame_number, 0) # Note currently not doing up/down stream recurl stations
frameHitPlotting(frame_hits_data, 4, frame_number, 0)
# frameHitPlotting(frame_hits_data, 3, frame_number, 1)
# frameHitPlotting(frame_hits_data, 3, frame_number, 2)
# frameHitPlotting(frame_hits_data, 4, frame_number, 1)
# frameHitPlotting(frame_hits_data, 4, frame_number, 2)
frameHitPlotting(frame_hits_data, 3, frame_number, "all")
frameHitPlotting(frame_hits_data, 4, frame_number, "all")
# layerHistPlottingSlow(frame_hits_data)
# layerHistPlottingFast(frame_hits_data)
# layerHistPlottingSlow(frame_hits_data)