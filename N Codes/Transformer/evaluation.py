import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def MomentumAccuracy(file):
    """Function to plot the momentum accuracy of the reconstruction.
    Input: 
    Reconstruction momenta
    Mc Real momenta
    Return:
    Graph of recon - mc momenta"""
    segs_data = pd.read_csv(file)
    segs_data["p"] = segs_data["p"].abs()
    segs_data["mc_p"] = segs_data["mc_p"].abs()
    momenta_difference = segs_data["p"] - segs_data["mc_p"]
    momenta_difference_range = momenta_difference[(momenta_difference >= -2) & (momenta_difference <= 2)]

    #Statistics
    Mean = np.mean(momenta_difference_range)
    S_dev = np.std(momenta_difference_range)
    print("Mean:", Mean)
    print("Standard deviation:" , S_dev)

    # Plot 
    bin_width = 0.05
    bins = np.arange(-2,2 , bin_width)

    plt.figure(figsize=(8, 6))
    plt.hist(momenta_difference, bins=bins, color='g', histtype='step', linewidth=2)
    #plt.gca().set_yticklabels([f"{int(tick/1e3)}" for tick in plt.gca().get_yticks()]) 
    plt.xlabel("$p_{\mathrm{rec}} - p_{\mathrm{mc}}$ (MeV/c)", fontsize = 14)
    plt.ylabel("Entries (per 0.05Mev/c)", fontsize = 14)
    plt.xlim(-2,2)
    plt.tick_params(axis='x', labelsize=14) 
    plt.tick_params(axis='y', labelsize=14) 

    plt.title("Momentum Accuracy", fontsize = 16)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Show plot
    plt.show()


file = "/root/Mu3eProject/RawData/TrirecFiles/signal1_99_32652/trirec_data_signal1_99_32652_frames.csv"
MomentumAccuracy(file)




