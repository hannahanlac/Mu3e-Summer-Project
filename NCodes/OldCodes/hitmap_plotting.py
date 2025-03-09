import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

def OneDIntensity(frame_hits_data, layer, station, divs_per_chip): 
    """Function to plot the 1D Intensity across the layers"""
    n = int(layer)
    layer_n_hits = frame_hits_data[frame_hits_data['layer'] == n]

    # Specifying the size of the different layers
    if layer == 1:
        chip_max = 6  # No. Pixel chips per layer layer 1 TDR
    
    elif layer == 2:
        chip_max = 6  # No. Pixel chips per layer layer 2 TDR

    elif layer == 3:
        chip_max = 17
        # NOTE: For 3 and 4 need to check the ladder numbers are correct for JUST the central barrel.

    elif layer == 4:
        chip_max = 18

    total_chips_central = chip_max
    total_chips = 3 * chip_max  # Upstream + Central + Downstream

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

    # Pixel sizes 
    pixel_size_y = 250 # Checked in meeting and this correct
    pixel_size_x = 256 

    # Define absolute hit positions
    if station in [0, 1, 2]:  # Single station case, no offset
        absolute_x_positions = layer_n_hits['pixel_x'] + (layer_n_hits['chip'] - 1) * pixel_size_x

    else:  # Combined stations case, apply offset
        absolute_x_positions = layer_n_hits['pixel_x'] + ((layer_n_hits['chip'] - 1) + chip_offset) * pixel_size_x

    # Define bins 
    bin_size_x = pixel_size_x / divs_per_chip # Change divs_per_chip to change the number of divisions in data per chip
    bins = int(chip_max * pixel_size_x / bin_size_x)

    #Define range for plotting
    x_min = 0
    x_max = chip_max * pixel_size_x


    #histogram
    intensity, edges = np.histogram(absolute_x_positions, bins=bins, range=(x_min, x_max))

    #intensity map
    plt.figure(figsize=(14, 6))
    plt.bar(edges[:-1], intensity, width=bin_size_x, align='edge', color='orange', edgecolor='black')

    plt.xlabel("Chip Number", fontsize=18)
    plt.ylabel("Number of Hits", fontsize=18)
    plt.title(f"1D Intensity Map for Layer {layer}, Station {station}", fontsize=20)

    # Set x-ticks to chip boundaries
    chip_ticks = [i * pixel_size_x for i in range(chip_max + 1)]
    plt.xticks(
        ticks=chip_ticks,
        labels=[f"{i}" for i in range(1, chip_max + 2)],
        fontsize=14,
        rotation=45
    )
    plt.tick_params(axis='y', labelsize=14) 

    # Add gridlines for better visualization
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    #Checking
    print("Total hits in data:", len(layer_n_hits))
    print("Total hits in histogram:", intensity.sum())



def frameHitPlotting (frame_hits_data, layer, frame_number, station):
    """Function for plotting hitmaps for each frame. 
    Input: Panda of hitmap data for given frame
    Output: Hitmaps for each layer""" 
    ##################### Outlining the sizes of the detector layers, and what to do on calling different layers ##############
    n = int(layer)
    layer_n_hits = frame_hits_data[frame_hits_data['layer'] == n]

    # Specifying the size of the different layers
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

    total_chips_central = chip_max
    total_chips = 3 * chip_max  # Upstream + Central + Downstream


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

    
    # Pixel sizes 
    pixel_size_y = 250 # Checked in meeting and this correct
    pixel_size_x = 256 
    
    ############################### Sorting the hits ########################
    # Define absolute hit positions
    if station in [0, 1, 2]:  # Single station case, no offset
        hit_x_positions_absolute = layer_n_hits['pixel_x'] + (layer_n_hits['chip'] - 1) * pixel_size_x

    else:  # Combined stations case, apply offset
        hit_x_positions_absolute = layer_n_hits['pixel_x'] + ((layer_n_hits['chip'] - 1) + chip_offset) * pixel_size_x

    hit_y_positions_absolute = layer_n_hits['pixel_y'] + (layer_n_hits['ladder'] - 1) * pixel_size_y
    
    
    
    # heatmap bin sizes - again need to change a bit as gpt helped
    bin_size_x = pixel_size_x /10
    bin_size_y = pixel_size_y /10
    x_bins = int(chip_max * pixel_size_x / bin_size_x)
    y_bins = int(ladder_max * pixel_size_y / bin_size_y)


    #heatmap 
    heatmap, xedges, yedges = np.histogram2d(
    hit_x_positions_absolute, hit_y_positions_absolute, bins=[x_bins, y_bins]
    )





     # Define the figure and plot
    # plt.figure(figsize=(10, 8))
    # #plt.scatter(hit_x_positions_absolute, hit_y_positions_absolute, color='red', s=20, label='Hit') # Scatter from doing 1 frame at time
    # plt.imshow(heatmap.T, cmap='hot', interpolation='nearest')

    # Working code for heatmap: Need to update as some GPT helped
    plt.figure(figsize=(14, 8))
    heatmap_plot =plt.imshow(
        heatmap.T, 
        cmap='hot', 
        interpolation='nearest', 
        origin='lower',  # To align with physical positions
        extent=[0, chip_max * pixel_size_x, 0, ladder_max * pixel_size_y])  # Match the physical dimensions

    cbar = plt.colorbar(heatmap_plot)
    cbar.set_label('Hit Count', fontsize=12)  # Add label to the color bar



    # Set the axis limits
    plt.xlim(0, chip_max * pixel_size_x)
    plt.ylim(0, ladder_max * pixel_size_y)

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

    # Add a grid for better visualization
    #plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show the plot
    plt.legend()
    plt.show()
 
# Specify range of frames plots wanted:
#frame_numbers = list(range(1,3))




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


file_name = "/root/Mu3eProject/WorkingVersion/Mu3eProject/DataFilesV5.3/signal1_96_32652/hits_data_signal1_96_32652_optimised_code.csv"
frame_hits_data = pd.read_csv(file_name)

frame_number = 1 # This is superfluous but will use to get working (as now going over whole file)
# frameHitPlotting(frame_hits_data, 1, frame_number, 0)
# frameHitPlotting(frame_hits_data, 2, frame_number, 0)
# frameHitPlotting(frame_hits_data, 3, frame_number, 0) # Note currently not doing up/down stream recurl stations
# frameHitPlotting(frame_hits_data, 4, frame_number, 0)
# frameHitPlotting(frame_hits_data, 3, frame_number, 1)
# frameHitPlotting(frame_hits_data, 3, frame_number, 2)
# frameHitPlotting(frame_hits_data, 4, frame_number, 1)
# frameHitPlotting(frame_hits_data, 4, frame_number, 2)
# frameHitPlotting(frame_hits_data, 1, frame_number, "all")
# frameHitPlotting(frame_hits_data, 1, frame_number, 0)
# frameHitPlotting(frame_hits_data, 1,frame_number, 2)




OneDIntensity(frame_hits_data, layer = 4, station = 'all', divs_per_chip = 4)






#layerHistPlottingSlow(frame_hits_data)
layerHistPlottingFast(frame_hits_data)


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