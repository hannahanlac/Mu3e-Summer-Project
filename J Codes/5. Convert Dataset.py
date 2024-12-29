import numpy as np
import uproot
import awkward as ak
import awkward0
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#loading original dataset, extract classes we need, cut off particles by set up eta standard and sort by pT (Transverse Momentum) value
def Dataset(signal_dir):
    '''signal_dir: Path to signal data (particles of interest).
    bkg_dir: Path to background data (irrelevant or noise particles).
    eta_std: Maximum allowed pseudorapidity value.'''

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
def padding(phi,eta,pT,train_frame,train_eta,train_phi,test_pT,test_phi,test_eta):

    desired_length_1 = np.max(ak.num_tracks(train_frame))
    train_frame = ak.to_num_trackspy(ak.fill_none(ak.pad_none(train_frame,desired_length_1),pT))

    desired_length_2 = np.max(ak.num_tracks(train_eta))
    train_eta = ak.to_num_trackspy(ak.fill_none(ak.pad_none(train_eta,desired_length_2),eta))

    desired_length_3 = np.max(ak.num_tracks(train_phi))
    train_phi = ak.to_num_trackspy(ak.fill_none(ak.pad_none(train_phi,desired_length_3),phi))

    desired_length_4 = np.max(ak.num_tracks(test_pT))
    test_pT = ak.to_num_trackspy(ak.fill_none(ak.pad_none(test_pT,desired_length_4),pT))

    desired_length_5 = np.max(ak.num_tracks(test_eta))
    test_eta = ak.to_num_trackspy(ak.fill_none(ak.pad_none(test_eta,desired_length_5),eta))

    desired_length_6 = np.max(ak.num_tracks(test_phi))
    test_phi = ak.to_num_trackspy(ak.fill_none(ak.pad_none(test_phi,desired_length_6),phi))

    return train_frame,train_eta,train_phi,test_pT,test_phi,test_eta


#Normalization using Standard Score
def Normalization(train_frame,train_eta,train_phi,test_pT,test_phi,test_eta):
    # define standard scaler
    scaler = StandardScaler()
    # transform data
    train_frame = scaler.fit_transform(train_frame)
    train_eta = scaler.fit_transform(train_eta)
    train_phi = scaler.fit_transform(train_phi)

    test_pT = scaler.fit_transform(test_pT)
    test_eta = scaler.fit_transform(test_eta)
    test_phi = scaler.fit_transform(test_phi)
    
    num_tracks1 = int(len(train_eta))
    mid1 = int(num_tracks1*0.5)
    num_tracks2 = int(len(test_eta))
    mid2 = int(num_tracks2*0.5)

    train_label_1 = np.ones(mid1)
    train_label_0 = np.zeros(mid1)
    train_label = np.concatenate((train_label_1,train_label_0))
    train_label = np.reshape(train_label,(num_tracks1,-1))

    test_label_1 = np.ones(mid2)
    test_label_0 = np.zeros(mid2)
    test_label = np.concatenate((test_label_1,test_label_0))
    test_label = np.reshape(test_label,(num_tracks2,-1))

    return train_frame,train_eta,train_phi,test_pT,test_phi,test_eta,train_label,test_label




#Finally making the dataset
def MakeDataset():
    train_dir = "inputs/thanks_train.awkd"
    test_dir = "inputs/thanks_test.awkd"
    phi = 10
    eta = 10
    pT = 0
    split_ratio = 0.75
    signal_dir = "inputs/delphes_output.root"

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
    # Dataset(signal_dir,bkg_dir,eta_std)
    layer_array,pixely_array,pixelx_array = Dataset(signal_dir)
    
    print ("Converting...")
    # fromiter_convert(split_ratio,layer_array,pixely_array,pixelx_array,pT_array_0,eta_array_0,phi_array_0)
    train_frame,train_eta,train_phi,test_pT,test_phi,test_eta = fromiter_convert(split_ratio,layer_array,pixely_array,pixelx_array)

    print ("Padding...")
    # padding(phi,eta,pT,train_frame,train_eta,train_phi,test_pT,test_phi,test_eta)
    train_frame,train_eta,train_phi,test_pT,test_phi,test_eta = padding(phi,eta,pT,train_frame,train_eta,train_phi,test_pT,test_phi,test_eta)

    print ("Normalizing...")
    # Normalization(train_frame,train_eta,train_phi,test_pT,test_phi,test_eta)
    train_frame,train_eta,train_phi,test_pT,test_phi,test_eta,train_label,test_label = Normalization(train_frame,train_eta,train_phi,test_pT,test_phi,test_eta)
    
    print ("Saving...")
    awkward0.save(train_dir, {"label": train_label, "eta_array": train_eta,"phi_array": train_phi,"pT_array": train_frame}, mode="w")
    awkward0.save(test_dir, {"label": test_label, "eta_array": test_eta,"phi_array": test_phi,"pT_array": test_pT}, mode="w")
    print ("...Done")
MakeDataset()