import numpy as np
import uproot
import awkward as ak
import awkward0
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#loading original dataset, extract classes we need, cut off particles by set up pixelx standard and sort by pixely,layer,station,ladder,chip (Transverse Momentum) value
def Dataset(signal_dir):
    '''signal_dir: Path to signal data (particles of interest).
    bkg_dir: Path to background data (irrelevant or noise particles).
    pixelx_std: Maximum allowed pseudorapidity value.'''

    #load dataset
    signal = uproot.open(signal_dir)

    #extract spatial (layer, station, ladder, chip, pixel x, pixel y) classes from original dataset
    signal_branch = signal['Delphes;1']
    # Check this, I don't think csv is that complicated (i.e., multi-layered), need specify
    frame_array = signal_branch['Particle/Particle.PT'].array(library="ak")
    pixelx_array = signal_branch['Particle/Particle.Pixel x'].array(library="ak")
    pixely_array = signal_branch['Particle/Particle.Pixel y'].array(library="ak")
    layer_array = signal_branch['Particle/Particle.Layer'].array(library="ak")
    station_array = signal_branch['Particle/Particle.Station'].array(library="ak")
    ladder_array = signal_branch['Particle/Particle.Ladder'].array(library="ak")
    chip_array = signal_branch['Particle/Particle.Chip'].array(library="ak")
    true_trackID = signal_branch['Particle/Particle.True Track ID'].array(library="ak")
    
    #sort hits by frame 
    index_1 = ak.argsort(frame_array, ascending = False)
    
    frame_array = frame_array[index_1]
    pixelx_array = pixelx_array[index_1]
    pixely_array = pixely_array[index_1]
    layer_array = layer_array[index_1]
    station_array = station_array[index_1]
    ladder_array = ladder_array[index_1]
    chip_array = chip_array[index_1]
    true_trackID = true_trackID[index_1]
    

    return frame_array,pixelx_array,pixely_array,layer_array,station_array,ladder_array,chip_array,true_trackID

#split the whole dataset and make labels for training and testing
def fromiter_convert(arrays,split_ratio,true_trackID):

    unique_tracks = np.unique(true_trackID)
    np.random.shuffle(unique_tracks)

    num_tracks = int(len(unique_tracks))
    mid = int(num_tracks*split_ratio)

    train_trackIDs = unique_tracks[0:mid]
    test_trackIDs = unique_tracks[mid:num_tracks]
    #assigns track IDs from 0 -> mid to be in our training dataset
    #assigns track IDs from mid -> rest to be in our testing dataset

    train_mask = np.isin(arrays['true_trackID'], train_trackIDs)    
    test_mask = ~train_mask
    #Create boolean masks here where filter through our data
    #Checks each value of true_trackID to see if selected for train_trackIDs, if does returns boolean mask True for that hit
    #Otherwise false. ~train_mask creates inverse of training mask. Hits not in training set assigned to testing set.

    train_data = {key: value[train_mask] for key, value in arrays.items()}
    test_data = {key: value[test_mask] for key, value in arrays.items()}
    #build new dictionaries with desired training and testing split based off track ID

    return train_data,test_data



#pad the jagged array to regular array, you can change the pad value you want
def padding(padding_values,train_data,test_data):

    for key, value in train_data.items():
        desired_length_1 = np.max(ak.num_tracks(value))
        pad_value = padding_values[f"{key}_pad"]
        train_data[key] = ak.to_numpy(ak.fill_none(ak.pad_none(value, desired_length_1), pad_value))

    for key, value in test_data.items():
        desired_length_2 = np.max(ak.num_tracks(value))
        pad_value = padding_values[f"{key}_pad"]
        test_data[key] = ak.to_numpy(ak.fill_none(ak.pad_none(value, desired_length_2), pad_value))

    return train_data,test_data


#Normalization using Standard Score
def Normalization(train_data,test_data):
    # define standard scaler
    scaler = StandardScaler()

    # transform data
    for key, value in train_data.items():
        #value not being used from above, might just need train_data.keys and no value or .items
        if key == 'true_trackID':  # Skip normalization for this key
            continue
        train_data[key] = scaler.fit_transform(train_data[key])
    
    for key, value in test_data.items():
        if key == 'true_trackID':  # Skip normalization for this key
            continue
        test_data[key] = scaler.fit_transform(test_data[key])


    return train_data,test_data




#Finally making the dataset
def MakeDataset():
    train_dir = "inputs/thanks_train.awkd"
    test_dir = "inputs/thanks_test.awkd"

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

    split_ratio = 0.75
    signal_dir = "ProcessedData/signal1_96_32652/csv/hits_data_signal1_96_32652.csv"

    arrays = {
    'frame_array': frame_array,
    'pixelx_array': pixelx_array,
    'pixely_array': pixely_array,
    'layer_array': layer_array,
    'station_array': station_array,
    'ladder_array': ladder_array,
    'chip_array': chip_array,
    'true_trackID': true_trackID,
    }

    train_data, test_data = split_dataset(0.75, arrays)



    print ("Reading dataset...")
    # Dataset(signal_dir,bkg_dir,pixelx_std)
    layer_array,pixely_array,pixelx_array = Dataset(signal_dir)
    
    print ("Converting...")
    # fromiter_convert(split_ratio,layer_array,pixely_array,pixelx_array,pixely,layer,station,ladder,chip_array_0,pixelx_array_0,frame_array_0)
    train_frame,train_pixelx,train_frame,test_pixely,layer,station,ladder,chip,test_frame,test_pixelx = fromiter_convert(split_ratio,layer_array,pixely_array,pixelx_array)

    print ("Padding...")
    # padding(frame,pixelx,pixely,layer,station,ladder,chip,train_frame,train_pixelx,train_frame,test_pixely,layer,station,ladder,chip,test_frame,test_pixelx)
    train_frame,train_pixelx,train_frame,test_pixely,layer,station,ladder,chip,test_frame,test_pixelx = padding(frame,pixelx,pixely,layer,station,ladder,chip,train_frame,train_pixelx,train_frame,test_pixely,layer,station,ladder,chip,test_frame,test_pixelx)

    print ("Normalizing...")
    # Normalization(train_frame,train_pixelx,train_frame,test_pixely,layer,station,ladder,chip,test_frame,test_pixelx)
    train_frame,train_pixelx,train_frame,test_pixely,layer,station,ladder,chip,test_frame,test_pixelx,train_label,test_label = Normalization(train_frame,train_pixelx,train_frame,test_pixely,layer,station,ladder,chip,test_frame,test_pixelx)
    
    print ("Saving...")
    awkward0.save(train_dir, {"label": train_label, "pixelx_array": train_pixelx,"frame_array": train_frame,"pixely,layer,station,ladder,chip_array": train_frame}, mode="w")
    awkward0.save(test_dir, {"label": test_label, "pixelx_array": test_pixelx,"frame_array": test_frame,"pixely,layer,station,ladder,chip_array": test_pixely,layer,station,ladder,chip}, mode="w")
    print ("...Done")
MakeDataset()