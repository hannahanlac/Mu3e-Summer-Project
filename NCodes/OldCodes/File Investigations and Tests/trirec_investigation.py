import uproot
import awkward as ak



def TrirecParts (file, tree, frame_no):
    """Function for compiling a dictionary of the parts of the Trirec file related to the Frames tree. Entry based on frame no."""
    frame_tree = file[tree].arrays()
    frame_frame = ak.Array([frame_tree[frame_no]])
    tree_info = {
        'runId' : frame_frame["runId"], #run id
        'frameId' : frame_frame["frameId"], #event ID
	    'x0' : frame_frame["x0"], #
	    'y0' : frame_frame["y0"], # 
	    'z0' : frame_frame["z0"], #
	    't0' : frame_frame["t0"], #
	    'r': frame_frame["r"], #
        'p': frame_frame["p"], #
        'chi2': frame_frame["chi2"], 
        'tan01': frame_frame["tan01"], 
	    'lam01': frame_frame["lam01"], # 
        'nhit': frame_frame["nhit"], 
        'n_shared_hits': frame_frame["n_shared_hits"], #
	    'n_shared_segs': frame_frame["n_shared_segs"], #
	    'n': frame_frame["n"], #
        'n3': frame_frame["n3"], #
        'n4': frame_frame["n4"], # 
        'n6': frame_frame["n6"], #
        'n8': frame_frame["n8"], #
        #### MC info #####
        'mc_eventId': frame_frame["mc_eventId"], #
        'mc_prime': frame_frame["mc_prime"],
        'mc': frame_frame["mc"],
        'mc_tid': frame_frame["mc_tid"], #
        'mc_pid': frame_frame["mc_pid"], #
        'mc_mid': frame_frame["mc_mid"], #
        'mc_type': frame_frame["mc_type"],
        'mc_p': frame_frame["mc_p"],
        'mc_pt': frame_frame["mc_pt"],
        'mc_phi': frame_frame["mc_phi"],
        'mc_lam': frame_frame["mc_lam"],
        'mc_theta': frame_frame["mc_theta"],
        'mc_vx': frame_frame["mc_vx"],
        'mc_vy': frame_frame["mc_vy"],
        'mc_vz': frame_frame["mc_vz"],
        'mc_vr': frame_frame["mc_vr"],
        'mc_vt': frame_frame["mc_vt"],
        'mc_t0': frame_frame["mc_t0"],
        'mc_hid0': frame_frame["mc_hid0"],

    }
    return tree_info

def PrintInfo(file, tree_type, frameNo):
    print(tree_type, "info:")
    tree_info = TrirecParts(file, tree = tree_type, frame_no = frameNo)
    print("runId", tree_info['runId'])
    print("frameId", tree_info['frameId'])
    print("x0", tree_info['x0'])
    print("y0", tree_info['y0'])
    print("z0", tree_info['z0'])
    print("t0", tree_info['t0'])
    print("r", tree_info['r'])
    print("p", tree_info['p'])
    print("chi2", tree_info['chi2'])
    print("tan01", tree_info['tan01'])
    print("lam01", tree_info['lam01'])
    print("nhit", tree_info['nhit'])
    print("n_shared_hits", tree_info['n_shared_hits'])
    print("n_shared_segs", tree_info['n_shared_segs'])
    print("no. of segs", tree_info['n'])
    print("no. of 3segs", tree_info['n3'])
    print("no. of 4segs", tree_info['n4'])
    print("no. of 6segs", tree_info['n6'])
    print("no. of 8segs", tree_info['n8'])
    print()
    ### MC info
    print("MC info:")
    print("MC_tids",tree_info['mc_tid'])
    print("MC_pids",tree_info['mc_pid'])
    print("MC_mids",tree_info['mc_mid'])
    print("MC_type",tree_info['mc_type'])
    print("mc(hits in right place?)",tree_info['mc']),
    print()
    print("Momenta and angle info")
    print("MC_p",tree_info['mc_p'])
    print("MC_pt",tree_info['mc_pt'])
    print("MC_phi",tree_info['mc_phi'])
    print("MC_lam",tree_info['mc_lam'])
    print("MC_theta",tree_info['mc_theta'])
    print("MC_vx",tree_info['mc_vx'])
    print("MC_vy",tree_info['mc_vy'])
    print("MC_vz",tree_info['mc_vz'])
    print("MC_vr",tree_info['mc_vr'])
    print("MC_vt",tree_info['mc_vt'])
    print("MC_t0",tree_info['mc_t0'])
    print("MC_hid0",tree_info['mc_hid0'])
    



file = uproot.open("/root/Mu3eProject/RawData/TrirecFiles/signal1_99_32652_execution_1_run_num_561343_trirec.root")


PrintInfo(file, tree_type = "frames", frameNo = 0 )
print()
#PrintInfo(tree_type = "frames_mc")



# useless seg tree
# seg_tree = file["segs"].arrays()

# print(seg_tree.fields)
# print(seg_tree["chi2"])