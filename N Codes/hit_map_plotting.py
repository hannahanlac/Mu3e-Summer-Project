import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import hist 

# Plotting the hits data:

def frameHitPlotting (frame_hits_data, layer, frame_number):
    """Function for plotting hitmaps for each frame. 
    Input: Panda of hitmap data for given frame
    Output: Hitmaps for each layer""" #NOTE: Only currently works for layer 1 - 4 not for upstream/downstream stations
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

    pixel_size_y = 250 # Checked in meeting and this correct
    pixel_size_x = 256 
    
    # Really need to know the x and y max of these pixels! Think I can guess at 250 from the hitmap data Mark has given me?? 
    pixel_x_max = chip_max* pixel_size_x
    pixel_y_max = ladder_max * pixel_size_y 
    hit_x_positions_absolute = layer_n_hits['pixel_x'] + (layer_n_hits['chip']-1)*pixel_size_x
    hit_y_positions_absolute = layer_n_hits['pixel_y'] + (layer_n_hits['ladder']-1)*pixel_size_y
    
    print(hit_x_positions_absolute)
    print(hit_y_positions_absolute)
    
    # heatmap bin sizes - again need to change a bit as gpt helped
    bin_size_x = pixel_size_x / 15
    bin_size_y = pixel_size_y / 15
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
    plt.title(f"All frames Layer {n} hits heatmap", fontsize=16)

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



def layerHistPlottingSlow (frame_hits_data):
  """Function that plots histogram of layer hits"""
  #n = int(layer)
  layer_1_hits, layer_2_hits, layer_3_hits, layer_4_hits = (
     frame_hits_data[frame_hits_data['layer'] == 1], 
     frame_hits_data[frame_hits_data['layer'] == 2],
     frame_hits_data[frame_hits_data['layer'] == 3], 
     frame_hits_data[frame_hits_data['layer'] == 4]
  )
  length_1 = len(layer_1_hits)
  length_2 = len(layer_2_hits)
  length_3 = len(layer_3_hits)
  length_4 = len(layer_4_hits)
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

    # Set x-axis ticks to match layers
  plt.xticks(layers, labels=[f"Layer {layer}" for layer in layers], fontsize=12)

  # Show the plot
  plt.tight_layout()
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




file_name = "/root/Mu3eProject/WorkingVersion/Mu3eProject/v5.3/signal1_95_32652/hits_data_signal1_95_32652.csv"
frame_hits_data = pd.read_csv(file_name)

frame_number = 1 # This is superfluous but will use to get working (as now going over whole file)
frameHitPlotting(frame_hits_data, 1, frame_number)
frameHitPlotting(frame_hits_data, 2, frame_number)
frameHitPlotting(frame_hits_data, 3, frame_number) # Note currently not doing up/down stream recurl stations
frameHitPlotting(frame_hits_data, 4, frame_number)
#layerHistPlottingSlow(frame_hits_data)
layerHistPlottingFast(frame_hits_data)
layerHistPlottingSlow(frame_hits_data)