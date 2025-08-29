The ROOT files are stored in DICE, in the directory /cephfs/dice/projects/mu3e/jack_mounser/v5.3/signal1. You will need to use the files ending in 'sort.root' and copy a few of these to your own directory.

There are two versions of the GNN pipeline, 'GNN_Workflow1' and 'GNN_Workflow2'. 

In 1, the graph building step forms edges between all possible combinations of hits and passes cylindrical coordinates to the model. 
It is a Graph Isomorphism Network with Edge Features (GINE). The GINE layers perform message passing.

In 2, the graph building step forms edges by a k nearest neighbors (kNN) algorithm and passes cartesian coordinates to the model.
It is a Multi-Layer Perceptron, without message passing or updating node embeddings.

To run the pipeline, run the scripts in the numbered order in the folder.

The previous projects students used a kNN algorithm to build graphs, with k=5, however I realised this filtered out a significant proportion of genuine edges, instead of just fake edges. Increasing k includes more genuine edges, however doesn't filter out many fake edges. I tried to build another model which would help filter out fake edges for the graph building step, as done in this paper: https://arxiv.org/abs/2407.12119.
The idea is to train a model to put the hits into an embedding space, where likely connected hits and positioned close together and unlikely connected hits spaced apart. This means that in the next part of the pipeline, you can use a kNN algorithm and a minimum distance between hits to form edges in the graph. This will filter out some fake edges as they are too far apart. I didn't have much success with this model, but I have put my attempts in Embedding_MLP folder. 

My suggested next steps would be to try get the Embedding_MLP working, as this would reduce the class imbalance of the dataset and remove a large number of fake edges from the graph. Additionally, I would suggest doing a hyperparameter search using WandB and investigating different GNN architectures.
