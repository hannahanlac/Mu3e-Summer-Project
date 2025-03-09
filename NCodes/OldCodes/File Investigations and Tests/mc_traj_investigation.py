import uproot
import awkward as ak


def MCTree (file):
    """Function for outputting some of the key information for the mu3e_mchits tree"""
    mc_tree = file["mu3e_mchits;1"]
    print(mc_tree.keys())  # List all branches
    print(mc_tree.show())  # Summary of branches and their types

    det_branch = mc_tree["det"].arrays() # This now correctly calls the tid part
    print("det branch:")
    print(det_branch[45245])
    print("Length of branch:", len(det_branch))
    print()

    print("tid branch:")
    tid_branch = mc_tree['tid'].arrays()
    print(tid_branch[45245])
    print("Length of branch:",len(tid_branch))
    print()

    print("pdg branch:")
    pdg_branch = mc_tree['pdg'].arrays()
    print(pdg_branch[45245])
    print("Length of branch:",len(pdg_branch))
    print()

    print("hid branch:")
    hid_branch = mc_tree['hid'].arrays()
    print(hid_branch[45245])
    print("Length of branch:",len(hid_branch))
    print()

    print("hid_g branch:")
    hid_branch = mc_tree['hid_g'].arrays()
    print(hid_branch[45245])
    print("Length of branch:",len(hid_branch))
    print()

    # print("hid_g branch:")
    # hid_g_branch = mc_tree['hid_g'].arrays()
    # print(hid_g_branch)
    # print("Length of branch:",len(hid_g_branch))
    # print()

    # print("edep branch:")
    # edep_branch = mc_tree['edep'].arrays()
    # print(edep_branch)
    # print("Length of branch:",len(edep_branch))
    # print()

    # print("time branch:")
    # time_branch = mc_tree['time'].arrays()
    # print(time_branch)
    # print("Length of branch:",len(time_branch))
    # print()

def mu3eMCTreeParts (file, frame_number):
    """Function for compiling a dictionary of the parts of the mu3e tree related to MC truth information"""
    mu3e_tree = file["mu3e"].arrays()
    mu3eFrame = ak.Array([mu3e_tree[frame_number]])
    mc_info = {
        'Ntraj' : mu3eFrame["Ntrajectories"], #Number of truth trajectories (particles)
        'traj_ID' : mu3eFrame["traj_ID"], #Run-unique ID of the particle
	    'traj_mother' : mu3eFrame["traj_mother"], # Run-unique ID of the mother particle of the current particle
	    'traj_PID' : mu3eFrame["traj_PID"], #Particle ID (PDG encoding) of the particle
	    'traj_type' : mu3eFrame["traj_type"], #Type and source of the particle, see Numbering and naming schemes on wiki
	    'traj_time' : mu3eFrame["traj_time"], #Creation time of the particle relative to the frame start time 
	    'traj_vx': mu3eFrame["traj_vx"], #Creation point of the particle
        'traj_vy': mu3eFrame["traj_vy"], #Creation point of the particle
        'traj_vz': mu3eFrame["traj_vz"], #Creation point of the particle

	    'traj_px': mu3eFrame["traj_px"], #Momentum vector of the particle at creation'
        'traj_py': mu3eFrame["traj_py"], #Momentum vector of the particle at creation'
        'traj_pz': mu3eFrame["traj_pz"], #Momentum vector of the particle at creation'

	    'traj_fbhid': mu3eFrame["traj_fbhid"], #number of SciFi crossings
	    'traj_tlhid': mu3eFrame["traj_tlhid"], #number of passages through tile volume  
        'traj_edep_target': mu3eFrame["traj_edep_target"], #Energy deposited in the target by this particle
    }
    return mc_info


file = uproot.open("/root/Mu3eProject/RawData/v5.3/signal1_99_32652_execution_1_run_num_561343_sort.root")

MCTree(file)

# mc_info = mu3eMCTreeParts(file, 0)
# # Print the contents of the mc_info dictionary dynamically
# for key, value in mc_info.items():
#     print(f"{key}: {value}")

# print("Length of traj_ID:" ,len(mc_info['traj_ID'][0]))
# print("Length of traj_PID:" ,len(mc_info['traj_PID'][0]))
# print(ak.type(mc_info['traj_ID']))

