import numpy as np
import awkward as ak
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


#loading original dataset, extract classes we need, cut off particles by set up pixelx standard and sort by pixely,layer,station,ladder,chip (Transverse Momentum) value
def Dataset(signal_dir):
    '''signal_dir: Path to signal data (particles of interest)'''

    # Load the CSV into a pandas DataFrame
    dataframe = pd.read_csv(signal_dir)

    #extract spatial (layer, station, ladder, chip, pixel x, pixel y) classes from original dataset
    #and convert to awkward array
    signal_arrays = ak.Array({
        'frame_array': dataframe['frameNumber'].to_numpy(),
        'hit_ID' : dataframe['hitIndex'].to_numpy(),
        'layer_array' :  dataframe['layer'].to_numpy(),
        'station_array' : dataframe['station'].to_numpy(),
        'ladder_array' : dataframe['ladder'].to_numpy(),
        'chip_array' : dataframe['chip'].to_numpy(),
        'tid_array' : dataframe['tid'].to_numpy(),
        'gx' : dataframe['gx'].to_numpy(),
        'gy' : dataframe['gy'].to_numpy(),
        'gz' : dataframe['gz'].to_numpy(),
        'traj_p' : dataframe['traj_p'].to_numpy(),
        'traj_pt' : dataframe['traj_pt'].to_numpy(),
        'traj_lambda' : dataframe['traj_lambda'].to_numpy(),
        'traj_phi' : dataframe['traj_phi'].to_numpy(),
    })
    
    #sort hit arrays by frame 
    sorting_index = ak.argsort(signal_arrays['frame_array'], ascending = True)
    
    signal_arrays = signal_arrays[sorting_index]
    

    return signal_arrays

#split the whole dataset and make labels for training and testing
def fromiter_convert(split_ratio, signal_arrays):

    unique_tracks = np.unique(ak.to_numpy(signal_arrays['tid_array']))
    # Convert to NumPy array for shuffling
    np.random.shuffle(unique_tracks)

    num_tracks = int(len(unique_tracks))
    mid = int(num_tracks*split_ratio)

    train_trackIDs = unique_tracks[0:mid]
    test_trackIDs = unique_tracks[mid:num_tracks]
    #assigns track IDs from 0 -> mid to be in our training dataset
    #assigns track IDs from mid -> rest to be in our testing dataset

    train_mask = np.isin(signal_arrays['tid_array'], train_trackIDs)    
    test_mask = ~train_mask
    #Create boolean masks here where filter through our data
    #Checks each value of tid to see if selected for train_trackIDs, if does returns boolean mask True for that hit
    #Otherwise false. ~train_mask creates inverse of training mask. Hits not in training set assigned to testing set.

    train_data = {key: signal_arrays[key][train_mask] for key in signal_arrays.fields}
    test_data = {key: signal_arrays[key][test_mask] for key in signal_arrays.fields}

    #build new awkward arrays with desired training and testing split based off track ID in a dictionary

    # Convert dictionaries to Awkward Arrays
    train_data = ak.Array(train_data)
    test_data = ak.Array(test_data)

    return train_data,test_data



#pad the jagged array to regular array, you can change the pad value you want
def padding(padding_values,train_data,test_data):

    for key in train_data.fields:
        value = train_data[key]

        desired_length_1 = np.max(len(value))
        pad_value = padding_values[f"{key}"]

        train_data[key] = ak.to_numpy(ak.fill_none(ak.pad_none(value, desired_length_1, 0), pad_value))

    for key in test_data.fields:
        value = test_data[key]

        desired_length_2 = np.max(len(value))
        pad_value = padding_values[f"{key}"]

        test_data[key] = ak.to_numpy(ak.fill_none(ak.pad_none(value, desired_length_2, 0), pad_value))

    return train_data,test_data


#Normalization using Standard Score
def Normalization(train_data,test_data):
    # define standard scaler
    scaler = StandardScaler()

    # transform data
    for key in train_data.fields:

        train_array = ak.to_numpy(train_data[key])
        #value not being used from above, might just need train_data.keys and no value or .items
        if key == 'tid_array':  # Skip normalization for this key
            continue

        if key == 'frame_array':
            continue

        if key == 'hit_ID':
            continue

        if key == 'layer_array':
            continue

        # Reshape to 2D for scaler
        train_array = train_array.reshape(-1, 1)
        train_array = scaler.fit_transform(train_array)

        train_data[key] = ak.Array(train_array)

    
    for key in test_data.fields:

        test_array = ak.to_numpy(test_data[key])
        #test_array is the numpy array, converted from the specific awkward key
        #loops through frame_array, pixelx_array etc.

        if key == 'tid_array':  # Skip normalization for this key (after convert to numpy so no issues when converting back later)
            continue

        if key == 'frame_array':
            continue

        if key == 'hit_ID':
            continue

        if key == 'layer_array':
            continue

        test_array = test_array.reshape(-1, 1)
        test_array = scaler.transform(test_array)

        test_data[key] = ak.Array(test_array)

    return train_data,test_data



#def check_data_stats(data, description=""):
#    for key in data.fields:
#        print(f"{description} {key}: mean={np.mean(data[key]):.3f}, std={np.std(data[key]):.3f}, min={np.min(data[key])}, max={np.max(data[key])}")



#Finally making the dataset
def MakeDataset():
    train_dir = "GithubRepoLinux/ProcessedData/signal1_96_32652/train_data"
    test_dir = "Github RepoLinux/ProcessedData/signal1_96_32652/test_data"

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    train_file = os.path.join(train_dir, "train_data.parquet")
    if os.path.exists(train_file): # Deletes old version of file if present
        os.remove(train_file)

    test_file = os.path.join(test_dir, "test_data.parquet")
    if os.path.exists(test_file): # Deletes old version of file if present
        os.remove(test_file)

    padding_values = {
        "frame_array": 0,
        "hit_ID": 0,
        "layer_array": 0,
        "station_array": 0,
        "ladder_array": 0,
        "chip_array": 0,
        "tid_array": 0,
        "gx" : 0,
        "gy" : 0,
        "gz" : 0,
        "traj_p" : 0,
        "traj_pt" : 0,
        "traj_lambda" : 0,
        "traj_phi" : 0
    }
    #Dictionary of padding values for each parameter
    #can change these padding values for physical/practical reasons as wish

    signal_dir = "GithubRepoLinux/ProcessedData/signal1_96_32652/csv/new_hits_data_signal1_96_32652.csv"
    split_ratio = 0.75


    print ("Reading dataset...")
    # Dataset(signal_dir)
    signal_arrays = Dataset(signal_dir)
    
    print ("Converting...")
    # fromiter_convert(split_ratio, signal_arrays)
    train_data, test_data = fromiter_convert(split_ratio, signal_arrays)

    print ("Padding...")
    # padding(padding_values,train_data,test_data)
    train_data, test_data = padding(padding_values, train_data, test_data)

    print ("Normalizing...")
    # Normalization(train_data,test_data)
    train_data, test_data = Normalization(train_data, test_data)

    #print("Validating...")
    #check_data_stats(train_data, "Train")
    #check_data_stats(test_data, "Test")

    print ("Saving...")
    ak.to_parquet(train_data, train_file) 
    ak.to_parquet(test_data, test_file)

    print ("...Done")

MakeDataset()