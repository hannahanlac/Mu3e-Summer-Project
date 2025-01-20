import numpy as np
import awkward as ak
import awkward0
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
        'frame_array': dataframe['frame'].to_numpy(),
        'pixelx_array' : dataframe['pixelx'].to_numpy(),
        'pixely_array' : dataframe['pixely'].to_numpy(),
        'layer_array' :  dataframe['layer'].to_numpy(),
        'station_array' : dataframe['station'].to_numpy(),
        'ladder_array' : dataframe['ladder'].to_numpy(),
        'chip_array' : dataframe['chip'].to_numpy(),
        'tid_array' : dataframe['tid'].to_numpy(),
    })
    
    #sort hit arrays by frame 
    sorting_index = ak.argsort(signal_arrays['frame_array'], ascending = False)
    
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

        if ak.num(value).ndim == 1:  # Check if the array is flat
            continue  # Skip padding for flat arrays

        desired_length_1 = ak.max(ak.num(value))
        pad_value = padding_values[f"{key}_pad"]

        train_data[key] = ak.fill_none(ak.pad_none(value, desired_length_1), pad_value)

    for key in test_data.fields:
        value = test_data[key]

        if ak.num(value).ndim == 1:  # Check if the array is flat
            continue  # Skip padding for flat arrays

        desired_length_2 = ak.max(ak.num(value))
        pad_value = padding_values[f"{key}_pad"]

        test_data[key] = ak.fill_none(ak.pad_none(value, desired_length_2), pad_value)

    return train_data,test_data


#Normalization using Standard Score
def Normalization(train_data,test_data):
    # define standard scaler
    scaler = StandardScaler()

    # transform data
    for key, value in train_data.items():
        #value not being used from above, might just need train_data.keys and no value or .items
        if key == 'tid_array':  # Skip normalization for this key
            continue
        train_data[key] = scaler.fit_transform(train_data[key])
    
    for key, value in test_data.items():
        if key == 'tid_array':  # Skip normalization for this key
            continue
        test_data[key] = scaler.fit_transform(test_data[key])


    return train_data,test_data




#Finally making the dataset
def MakeDataset():
    train_dir = "ProcessedData/signal1_96_32652/train_data"
    test_dir = "ProcessedData/signal1_96_32652/test_data"

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
        "pixelx_array": 0,
        "pixely_array": 0,
        "layer_array": 0,
        "station_array": 0,
        "ladder_array": 0,
        "chip_array": 0,
    }
    #Dictionary of padding values for each parameter
    #can change these padding values for physical/practical reasons as wish

    signal_dir = "ProcessedData/signal1_96_32652/csv/hits_data_signal1_96_32652.csv"
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
    
    print ("Saving...")
    ak.to_parquet(train_data, train_file) 
    ak.to_parquet(test_data, test_file)

    print ("...Done")

MakeDataset()