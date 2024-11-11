import uproot
import numpy
import matplotlib.pyplot as plt

file = uproot.open("/root/Mu3eProject/RawData/HitData/signal1_1_1944629_execution_1_run_num_836827_sort.root")
hits1 = file["mu3e;1"].arrays()



#print(hits1.fields)

#print(hits1.Nhit.show)

keys = file.keys()
print("Keys in the ROOT file:", keys)