import numpy as np
import uproot
import awkward as ak
import awkward0
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#loading original dataset, extract classes we need, cut off particles by set up eta standard and sort by pT (Transverse Momentum) value
def Dataset(signal_dir,bkg_dir,eta_std):
    '''signal_dir: Path to signal data (particles of interest).
    bkg_dir: Path to background data (irrelevant or noise particles).
    eta_std: Maximum allowed pseudorapidity value.'''

    #load dataset
    signal = uproot.open(signal_dir)
    bkg = uproot.open(bkg_dir)

    #extract eta,pT.phi classes from original dataset
    signal_branch = signal['Delphes;1']
    phi_array_1 = signal_branch['Particle/Particle.Phi'].array(library="ak")
    eta_array_1 = signal_branch['Particle/Particle.Eta'].array(library="ak")
    pT_array_1 = signal_branch['Particle/Particle.PT'].array(library="ak")
    bkg_branch = bkg['Delphes;1']
    phi_array_0 = bkg_branch['Particle/Particle.Phi'].array(library="ak")
    eta_array_0 = bkg_branch['Particle/Particle.Eta'].array(library="ak")
    pT_array_0 = bkg_branch['Particle/Particle.PT'].array(library="ak")

    #cut off particles whose absolute eta value > eta_std in each event
    phi_array_1 = phi_array_1[(-eta_std < eta_array_1) & (eta_array_1 < eta_std)]
    pT_array_1 = pT_array_1[(-eta_std < eta_array_1) & (eta_array_1 < eta_std)]
    eta_array_1 = eta_array_1[(-eta_std < eta_array_1) & (eta_array_1 < eta_std)]
    
    phi_array_0 = phi_array_0[(-eta_std < eta_array_0) & (eta_array_0 < eta_std)]
    pT_array_0 = pT_array_0[(-eta_std < eta_array_0) & (eta_array_0 < eta_std)]
    eta_array_0 = eta_array_0[(-eta_std < eta_array_0) & (eta_array_0 < eta_std)]
    
    #sort particles by pT value 
    index_1 = ak.argsort(pT_array_1, ascending = False)
    
    pT_array_1 = pT_array_1[index_1]
    eta_array_1 = eta_array_1[index_1]
    phi_array_1 = phi_array_1[index_1]


    index_2 = ak.argsort(pT_array_0, ascending = False)

    pT_array_0 = pT_array_0[index_2]
    eta_array_0 = eta_array_0[index_2]
    phi_array_0 = phi_array_0[index_2]

    return pT_array_1,eta_array_1,phi_array_1,pT_array_0,eta_array_0,phi_array_0

#split the whole dataset and make labels for training and testing
def fromiter_convert(split_ratio,pT_array_1,eta_array_1,phi_array_1,pT_array_0,eta_array_0,phi_array_0):

    num = int(len(phi_array_1))
    mid = int(num*split_ratio)

    #training part
    a1 = awkward0.fromiter(pT_array_1[0:mid])
    b1 = awkward0.fromiter(eta_array_1[0:mid])
    c1 = awkward0.fromiter(phi_array_1[0:mid])
    a2 = awkward0.fromiter(pT_array_0[0:mid])
    b2 = awkward0.fromiter(eta_array_0[0:mid])
    c2 = awkward0.fromiter(phi_array_0[0:mid])
    #testing part
    a3 = awkward0.fromiter(pT_array_1[mid:num])
    b3 = awkward0.fromiter(eta_array_1[mid:num])
    c3 = awkward0.fromiter(phi_array_1[mid:num])
    a4 = awkward0.fromiter(pT_array_0[mid:num])
    b4 = awkward0.fromiter(eta_array_0[mid:num])
    c4 = awkward0.fromiter(phi_array_0[mid:num])

    #concatenate siganls and background together
    train_pT = ak.concatenate((a1,a2))
    train_eta = ak.concatenate((b1,b2))
    train_phi = ak.concatenate((c1,c2))

    test_pT = ak.concatenate((a3,a4))
    test_eta = ak.concatenate((b3,b4))
    test_phi = ak.concatenate((c3,c4))

    return train_pT,train_eta,train_phi,test_pT,test_phi,test_eta


#pad the jagged array to regular array, you can change the pad value you want
def padding(phi,eta,pT,train_pT,train_eta,train_phi,test_pT,test_phi,test_eta):

    desired_length_1 = np.max(ak.num(train_pT))
    train_pT = ak.to_numpy(ak.fill_none(ak.pad_none(train_pT,desired_length_1),pT))

    desired_length_2 = np.max(ak.num(train_eta))
    train_eta = ak.to_numpy(ak.fill_none(ak.pad_none(train_eta,desired_length_2),eta))

    desired_length_3 = np.max(ak.num(train_phi))
    train_phi = ak.to_numpy(ak.fill_none(ak.pad_none(train_phi,desired_length_3),phi))

    desired_length_4 = np.max(ak.num(test_pT))
    test_pT = ak.to_numpy(ak.fill_none(ak.pad_none(test_pT,desired_length_4),pT))

    desired_length_5 = np.max(ak.num(test_eta))
    test_eta = ak.to_numpy(ak.fill_none(ak.pad_none(test_eta,desired_length_5),eta))

    desired_length_6 = np.max(ak.num(test_phi))
    test_phi = ak.to_numpy(ak.fill_none(ak.pad_none(test_phi,desired_length_6),phi))

    return train_pT,train_eta,train_phi,test_pT,test_phi,test_eta


#Normalization using Standard Score
def Normalization(train_pT,train_eta,train_phi,test_pT,test_phi,test_eta):
    # define standard scaler
    scaler = StandardScaler()
    # transform data
    train_pT = scaler.fit_transform(train_pT)
    train_eta = scaler.fit_transform(train_eta)
    train_phi = scaler.fit_transform(train_phi)

    test_pT = scaler.fit_transform(test_pT)
    test_eta = scaler.fit_transform(test_eta)
    test_phi = scaler.fit_transform(test_phi)
    
    num1 = int(len(train_eta))
    mid1 = int(num1*0.5)
    num2 = int(len(test_eta))
    mid2 = int(num2*0.5)

    train_label_1 = np.ones(mid1)
    train_label_0 = np.zeros(mid1)
    train_label = np.concatenate((train_label_1,train_label_0))
    train_label = np.reshape(train_label,(num1,-1))

    test_label_1 = np.ones(mid2)
    test_label_0 = np.zeros(mid2)
    test_label = np.concatenate((test_label_1,test_label_0))
    test_label = np.reshape(test_label,(num2,-1))

    return train_pT,train_eta,train_phi,test_pT,test_phi,test_eta,train_label,test_label


#Finally making the dataset
def MakeDataset():
    train_dir = "inputs/thanks_train.awkd"
    test_dir = "inputs/thanks_test.awkd"
    phi = 10
    eta = 10
    pT = 0
    split_ratio = 0.75
    signal_dir = "inputs/delphes_output.root"
    bkg_dir = "inputs/delphes_nuGun.root"
    eta_std = 3.0

    print ("Reading dataset...")
    # Dataset(signal_dir,bkg_dir,eta_std)
    pT_array_1,eta_array_1,phi_array_1,pT_array_0,eta_array_0,phi_array_0 = Dataset(signal_dir,bkg_dir,eta_std)
    
    print ("Converting...")
    # fromiter_convert(split_ratio,pT_array_1,eta_array_1,phi_array_1,pT_array_0,eta_array_0,phi_array_0)
    train_pT,train_eta,train_phi,test_pT,test_phi,test_eta = fromiter_convert(split_ratio,pT_array_1,eta_array_1,phi_array_1,pT_array_0,eta_array_0,phi_array_0)

    print ("Padding...")
    # padding(phi,eta,pT,train_pT,train_eta,train_phi,test_pT,test_phi,test_eta)
    train_pT,train_eta,train_phi,test_pT,test_phi,test_eta = padding(phi,eta,pT,train_pT,train_eta,train_phi,test_pT,test_phi,test_eta)

    print ("Normalizing...")
    # Normalization(train_pT,train_eta,train_phi,test_pT,test_phi,test_eta)
    train_pT,train_eta,train_phi,test_pT,test_phi,test_eta,train_label,test_label = Normalization(train_pT,train_eta,train_phi,test_pT,test_phi,test_eta)
    
    print ("Saving...")
    awkward0.save(train_dir, {"label": train_label, "eta_array": train_eta,"phi_array": train_phi,"pT_array": train_pT}, mode="w")
    awkward0.save(test_dir, {"label": test_label, "eta_array": test_eta,"phi_array": test_phi,"pT_array": test_pT}, mode="w")
    print ("...Done")
MakeDataset()