# Notes on running GNN workflow

### Setup

Existing inputs are stored here: `/hdfs/user/ec6821/BristolFastML/Samples/GNN_Russell/`

Copy them locally:
```
mkdir inputs
hadoop fs -copyToLocal /user/ec6821/BristolFastML/Samples/GNN_Russell/* inputs/
```

These inputs are from DELPHES, not the CMSSW ones used by default so far.  They represent particles we have offline, and the performance we see with a model trained on these inputs is very likely to be better than what we would see with CMSSW inputs.

### First step - 'convert' dataset

Reads the datasets, removes particles outside the eta acceptance, sort by pt.
Then arrange into training and testsing sets, assign signal/background labels.
Pad events with dummy particles, so that all events have the same number of inputs.
Normalize eta, phi, pt distributions.

Takes ~10 mins (converting step takes the longest).
```
python step1_convert_dataset.py 
```

Outputs are in `inputs/` directory (`thanks_train/test.awkd`).

### Second step - train model

Notebook : `step2_ParticleNetLite++.ipynb`

Couldn't open this notebook, but could convert it to a regular python script : `step2_ParticleNetLite++.py`
This python script has been updated to not point at particular user directories.

The scripts trains the model over 10 epochs, taking ~1min per epoch:
```
python step2_ParticleNetLite++.py
```
The trained model ends up in `model_record/<model_name>/`, where `model_name` is specified in the script.

### Third step - make plots

Make some plots to analyze performance of the model that was trained in the previous step.

`model_name` is specified again in this script, so if you change it in the previous script, you need to update it here.

```
python step3_make_plots2.py 
```

Plots end up in `model_record/<model_name>/`.
