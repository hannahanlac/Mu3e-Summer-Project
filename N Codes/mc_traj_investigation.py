import uproot
import awkward


file = uproot.open("/root/Mu3eProject/RawData/v5.3/signal1_95_32652_execution_1_run_num_67021_sort.root")

mc_tree1 = file["mu3e_mchits;1"].arrays
mc_tree2 = file["mu3e_mchits;2"].arrays


tree = file["mu3e_mchits;1"]
print(tree.keys())  # Lists all branches
print(tree.show())  # Provides a summary of branches and their types

tid_tree = tree["det"].arrays() # This now correctly calls the tid part
tid_1 = tid_tree[0]
print(tid_1)





mu3eTree = file["mu3e"].arrays() #This is my current best guess at what the hit information is 
print(mu3eTree.fields)

# frame_1 = mu3eTree[1]
# frame_1_mc = frame_1["det"]
# print(frame_1_mc) # Finding the fields in hit info: Does appear to be for hit info
#print(len(frame_1_mc))
# print(mu3eTree) # Printing the first part of hits1: All of the hit information
# print()

# # mc_hits = file["mu3e_mchits"].arrays
# # print(mc_hits["traj"])
# # print()


