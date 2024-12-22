# %%
import numpy as np
import uproot
import awkward as ak
import awkward0

# %%
signal = uproot.open("/software/dg22882/ParticleNet/Dataset/HH/ROOT/delphes_output.root") # change path to origial dataset
bkg = uproot.open("/software/dg22882/ParticleNet/Dataset/SNU/ROOT/delphes_nuGun.root") # change path to origial dataset

# %%
signal_branch = signal['Delphes;1']
phi_array_1 = signal_branch['Particle/Particle.Phi'].array(library="ak")
eta_array_1 = signal_branch['Particle/Particle.Eta'].array(library="ak")
pT_array_1 = signal_branch['Particle/Particle.PT'].array(library="ak")
bkg_branch = bkg['Delphes;1']
phi_array_0 = bkg_branch['Particle/Particle.Phi'].array(library="ak")
eta_array_0 = bkg_branch['Particle/Particle.Eta'].array(library="ak")
pT_array_0 = bkg_branch['Particle/Particle.PT'].array(library="ak")

# %%
a1 = awkward0.fromiter(pT_array_1)
b1 = awkward0.fromiter(eta_array_1)
c1 = awkward0.fromiter(phi_array_1)
a2 = awkward0.fromiter(pT_array_0)
b2 = awkward0.fromiter(eta_array_0)
c2 = awkward0.fromiter(phi_array_0)

# %%
for event in range(len(a1)):
    ind = np.where(a1[event]>255)
    if np.shape(ind[0]) != 0:
        a1[event][ind] = 255

pT1 = a1 / 255

# %%
for event in range(len(a2)):
    ind = np.where(a2[event]>255)
    if np.shape(ind[0]) != 0:
        a2[event][ind] = 255

pT0 = a2 / 255

# %%
print(type(a1),type(a2),type(pT1),type(pT0))

# %%
train_pT = ak.concatenate((pT1,pT0))
train_eta = ak.concatenate((b1,b2))
train_phi = ak.concatenate((c1,c2))

# %%
label_10 = np.concatenate((np.ones(100000),np.zeros(100000)))
label_01 = np.concatenate((np.zeros(100000),np.ones(100000)))

train_label = np.concatenate([label_10[:,None],label_01[:,None]],axis=1)

# %%
train_pT = ak.sort(train_pT,ascending = False, highlevel= True)
train_eta = ak.sort(train_eta,ascending = False, highlevel= True)
train_phi = ak.sort(train_phi,ascending = False, highlevel= True)

# %%
train_pT_100 = train_pT[:,:100]
train_eta_100 = train_eta[:,:100]
train_phi_100 = train_phi[:,:100]

# %%
train_pT_100

# %%
desired_length_1 = np.max(ak.num(train_pT_100))
train_pT_100 = ak.to_numpy(ak.pad_none(train_pT_100,desired_length_1))

desired_length_2 = np.max(ak.num(train_eta_100))
train_eta_100 = ak.to_numpy(ak.pad_none(train_eta_100,desired_length_2))

desired_length_3 = np.max(ak.num(train_phi_100))
train_phi_100 = ak.to_numpy(ak.pad_none(train_phi_100,desired_length_3))

# %%
train_label = ak.to_numpy(train_label)

# %%
awkward0.save("/software/dg22882/ParticleNet/Dataset/train_100_200000.awkd", {"label": train_label, "eta_array": train_eta_100,"phi_array": train_phi_100,"pT_array": train_pT_100}, mode="w")


