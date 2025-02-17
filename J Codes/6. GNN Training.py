from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import awkward as ak
import awkward0
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.python import keras
import os

print(tf.__version__)
print(dir(tf.keras))
print(hasattr(tf.keras, "layers"))  # Should return True


import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


# ### Adjust train_dataset

# In[7]:

class Dataset(object):
    
    """Dataset class to load and preprocess data from a Parquet file.

    Args:
        filepath (str): Path to the Parquet dataset.
        feature_dict (dict, optional): Dictionary mapping feature groups to feature names.
        label (str, optional): The target variable (default: 'tid_array').
    data_format (str, optional): Feature array structure ('channel_first' or 'channel_last')."""

    def __init__(self, filepath, feature_dict = None, label='tid_array', data_format='channel_first'):

        assert data_format in ('channel_first', 'channel_last'), "Invalid data format"
        #Ensures data_format is either 'channel_first' or 'channel_last'. If not, we've added error message to tell us this


        #self - convention in classes to refer to any proccessing corresponding to the current instance being worked with
        #filepath: Path to the Parquet dataset. feature_dict: Dictionary mapping feature groups to feature names. Defaults/Starts to/with None
        #label: The target variable (default: 'tid_array'). data_format: Determines how feature arrays are structured ('channel_first' or 'channel_last').
        self.filepath = filepath
        self.label = label

        self.feature_dict = feature_dict if feature_dict is not None else {}
        #If feature_dict is not None, use its given value. Else, use a new empty dictionary {}.
        if not self.feature_dict:
            #If feature_dict is empty (which it should be), we initialize default feature groups.
            #This syntax checks whether empty directly instead of counting as ==0 method did before
            self.feature_dict['points'] = ['pixelx_array', 'pixely_array', 'layer_array', 'station_array', 'ladder_array', 'chip_array']
            self.feature_dict['features'] = ['pixelx_array', 'pixely_array', 'layer_array', 'station_array', 'ladder_array', 'chip_array']
            self.feature_dict['mask'] = ['layer_array']
            #"pixelx_array": b,"pixely_array": c,"layer_array"

        
        self.stack_axis = 1 if data_format == 'channel_first' else -1
        #Sets the axis for stacking features: 1 -> Features will be stacked along channels (channel_first).
        #-1 -> Features will be stacked along last dimension (channel_last).
        self._values = {}
        self._label = None

        self._load()


    def _load(self):
        """Loads the dataset from a Parquet file and performs integrity checks."""

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset file '{self.filepath}' not found.")

        logging.info('Start loading file %s' % self.filepath)
        #logs a message indicating that the loading process has started - useful for debugging later

        a = ak.from_parquet(self.filepath)
        #Loads a dataset from a Parquet file into an Awkward Array (a)
        #print(ak.to_list(a[:5]))  # Prints first 5 entries in list 

        if len(a) == 0:
            raise ValueError("Loaded dataset is empty!")

        # Check if label exists in the dataset
        if self.label not in a.fields:
            raise KeyError(f"Label '{self.label}' not found in dataset.")
        #Prevents errors later by ensuring the label column exists in the dataset now, at this stage
        #Preemptive error handling suggested by GPT
            
        # Store label separately
        self._label = ak.to_numpy(a[self.label])  # Convert label to NumPy array
        #Converts Awkward Array to NumPy array for easier processing.
        #Extracts the target label column for training (tid_array)
        #Extracts the label column from the dataset (a)
        #extracted data is stored in self._label for later use
        #could be issue here, how determining where / what label is?
        print(f"Label '{self.label}' shape: {self._label.shape}")

        if self._label.ndim != 1:
            raise ValueError(f"Label array must be 1D, but got shape {self._label.shape}")

        self._values = {}
        # Initialize dictionary to store feature arrays

        counts = None
        #This variable will store the number of elements in the first feature column that is processed
        #used to ensure that all feature columns have the same number of elements
  

        for feature_group, columns in self.feature_dict.items():
            #Iterates over feature groups and their associated columns.
            feature_arrays = []
            
            for col in columns:
                if col not in a.fields:
                    logging.warning(f"Column '{col}' not found in dataset. Skipping.")
                    continue  # Skip missing columns  
            #some columns might not exist in all datasets.
            #Instead of crashing, it logs a warning and skips the missing column.

                feature_array = ak.to_numpy(a[col])
                #Converts each feature column from Awkward to NumPy.

                # Ensure consistent feature lengths
                feature_length = len(feature_array)
                if counts is None:
                    counts = feature_length
                else:
                    assert counts == feature_length, f"Inconsistent feature lengths in column '{col}'! Expected {counts}, got {feature_length}."
                #How works: First column determines counts, and all subsequent columns must match.
                #Otherwise logs that error

                # Flatten single-element lists into scalars
                if feature_array.ndim == 2 and feature_array.shape[1] == 1:
                    feature_array = feature_array.flatten()
                    #Your dataset has lists inside lists e.g., 'pixelx_array': [[-0.7253]], 'pixely_array': [[1.1884]]
                    #.flatten() converts [[-0.7253]] -> [-0.7253], ensuring proper shape.

                feature_arrays.append(feature_array)
                #append it now that  all checks and balances have been passed
            
            
            if feature_arrays:
                self._values[feature_group] = np.stack(feature_arrays, axis=self.stack_axis)
            #Stacks features along the chosen axis (stack_axis).
                print(f"Feature Group '{feature_group}' shape: {self._values[feature_group].shape}")

            else:
                logging.warning(f"Feature group '{feature_group}' has no valid columns.")
            #Handles empty feature groups (e.g., if all columns were missing).


        logging.info('Finished loading file %s' % self.filepath)
        #Logs a message indicating that the dataset has been fully loaded


    def __len__(self):
        return len(self._label)
        #Returns the number of samples in the dataset.

    def __getitem__(self, key):
        if key==self.label:
            return self._label
        elif key in self._values:
            return self._values[key]
        else:
            raise KeyError(f"Feature '{key}' not found in dataset.")
            #Allows indexing like a dictionary (dataset['pixelx_array'] returns that feature).
    
    @property
    def X(self):
        return self._values
    
    @property
    def y(self):
        return self._label

    #Encapsulates feature (X) and label (y) access.

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        shuffle_indices = np.arange(len(self))
        np.random.shuffle(shuffle_indices)

        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]

        self._label = self._label[shuffle_indices]
    #Randomly shuffles the dataset.
    #Uses consistent shuffling for features and labels.
    #Need to ensure no index mixxing happening here





# In[2]:

def Batching(frames, hits_dict, labels_dict, frames_per_batch):
    """
    Groups hit points by frame number and creates batches of frames for training.

    Parameters:
    - frames (list): List of unique frame numbers.
    - hits_dict (dict): Dictionary mapping frame numbers to hit feature tensors (P, C_f).
    - labels_dict (dict): Dictionary mapping frame numbers to label tensors (P,).
    - frames_per_batch (int): Number of frames to include in each batch.

    Returns:
    - A tf.data.Dataset containing batches of (X, y), where:
      - X has shape (frames_per_batch, P, C_f)
      - y has shape (frames_per_batch, P)
    """

    # Ensure frames_per_batch is valid
    if not isinstance(frames_per_batch, int) or frames_per_batch <= 0:
        raise ValueError(f"frames_per_batch must be a positive integer, got {frames_per_batch}")

    # Ensure there are enough frames for at least one batch
    if len(frames) < frames_per_batch:
        raise ValueError(f"Number of frames ({len(frames)}) is less than frames_per_batch ({frames_per_batch})")


    batched_features = []  # Stores batches of hit features
    batched_labels = []    # Stores batches of labels

    #Check frame consistency
    missing_frames = [frame for frame in frames if frame not in hits_dict or frame not in labels_dict]
    if missing_frames:
        raise KeyError(f"Some frames are missing from hits_dict or labels_dict: {missing_frames}")

    # Verify tensor shapes
    first_frame = frames[0]
    expected_feature_shape = hits_dict[first_frame].shape  # Expect (P, C_f)
    expected_label_shape = labels_dict[first_frame].shape  # Expect (P,)

    for frame in frames:
        hit_shape = hits_dict[frame].shape
        label_shape = labels_dict[frame].shape

        if hit_shape != expected_feature_shape:
            raise ValueError(f"Inconsistent feature shape for frame {frame}. Expected {expected_feature_shape}, got {hit_shape}")
        
        if label_shape != expected_label_shape:
            raise ValueError(f"Inconsistent label shape for frame {frame}. Expected {expected_label_shape}, got {label_shape}")

    # Loop over frames in steps of frames_per_batch
    for i in range(0, len(frames), frames_per_batch):
        batch_frames = frames[i:i+frames_per_batch]  # Select frames_per_batch frames

        # Ensure batch is full (handle last batch case)
        if len(batch_frames) < frames_per_batch:
            continue  # Skip incomplete batch

        print(f"Processing batch {i//frames_per_batch + 1} with {len(batch_frames)} frames")  # Check batch count

        # Extract hit features and labels for these selected frames
        batch_hits = np.stack([hits_dict[frame] for frame in batch_frames])  # Shape: (frames_per_batch, P, C_f)
        batch_labels = np.stack([labels_dict[frame] for frame in batch_frames])  # Shape: (frames_per_batch, P)

        #Append these stacked features into this batch's dictionary of values and labels
        batched_features.append(batch_hits)
        batched_labels.append(batch_labels)

    # Convert lists to TensorFlow tensors
    batched_features = tf.convert_to_tensor(batched_features, dtype=tf.float32)
    batched_labels = tf.convert_to_tensor(batched_labels, dtype=tf.int32)

    print(f"Stacked features shape: {batched_features[-1].shape}")  # Check shape after stacking
    print(f"Stacked labels shape: {batched_labels[-1].shape}")

    # Convert to a TensorFlow dataset
    batched_dataset = tf.data.Dataset.from_tensor_slices((batched_features, batched_labels))

    # Shuffle before batching for proper training
    batched_dataset = batched_dataset.shuffle(buffer_size=len(batched_features)).batch(1) # 1 batch = 1 group of frames_per_batch

    return batched_dataset




#Calculates euclidean distances in graph space between nodes, knn will use this output to determine
#what counts as being nn. Should be able to stay as is, general enough.
def batch_distance_matrix_general(A, B):

    with tf.name_scope('dmat'):

        #Compute squared norms
        r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
        r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)

        #Compute distance matrix
        m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
        D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))

        print(r_A)
        print(r_B)
        print(m)
        
        print(D)
        print(f"Distance matrix shape: {D.shape}")
        print(f"Min distance: {tf.reduce_min(D)}, Max distance: {tf.reduce_max(D)}")

        return D



# ### k-nearest neighbors
# construct graph data using knn algorithm
#collects the knn based on results from above? Or based on something else? I don't see D in here anywhere
def knn(num_points, k, topk_indices, features):
    print(f"Features shape: {features.shape}")
    # topk_indices: (N, P, K)
    # features: (N, P, C)
    with tf.name_scope('knn'):
        queries_shape = tf.shape(features)
        batch_size = queries_shape[0]
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(topk_indices, axis=3)], axis=3)  # (N, P, K, 2)

        # Gather neighbor features, preserving C_f dimension
        knn_features = tf.gather_nd(features, indices)  # (N, P, K, C_f)

        # Ensure correct shape
        knn_features = tf.ensure_shape(knn_features, [None, None, k, None])

        return knn_features
    



# ### Edge Convulution operation
# Attention: use (1,1) kernel Conv2D to perform MLP

# In[3]:

#main message passing operation of GNN can make major theory tweaks here
def edge_conv(points, features, num_points, K, channels, with_bn=True, activation='relu', name='edgeconv'):
    """Modified EdgeConv for Edge Classification
    
    Args:
        points: (N, P, C_p) - Hit positions (N events/sampels, P hits, C_p position features)
        features: (N, P, C_f) - Hit features (N events/samples, P hits, C_f feature channels)
        num_points: Number of hits per event/sample(batch)
        K: Number of nearest neighbors (int)
        channels: Tuple of MLP output sizes
        with_bn: Whether to apply batch normalization
        activation: Activation function (default: 'relu')
        name: Name scope for layers
        pooling: pooling method ('max' or 'average')
    
    Returns:
        edge_logits: (N, P, K, 1) - Binary classification for each edge
    """
    

    with tf.name_scope(name='edgeconv'):

        # Compute kNN graph
        D = batch_distance_matrix_general(points, points) # (N, P, P), Pairwise distances between nodes in graph space?
        _, indices = tf.nn.top_k(-D, k=K + 1)  # (N, P, K+1), collects K+1 nearest neighbors of each node (including self)
        indices = indices[:, :, 1:]  # (N, P, K) removes the self-connection so that only the actual K-nearest neighbors remain

        # Get neighbor features
        knn_features = knn(num_points, K, indices, features)  # (N, P, K, C_f) 
        #extracts features of the K nearest neighbors for each hit and stores by calling knn function 

        print(f"KNN Features shape: {features.shape}")

        knn_features_center = tf.tile(tf.expand_dims(features, axis=2), (1, 1, K, 1))  # (N, P, K, C_f)


        #duplicates the central hit’s features (i.e., hit focusing on currently) so can compare with neighbors
        edge_features = tf.concat([knn_features_center, tf.subtract(knn_features, knn_features_center)], axis=-1)  # (N, P, K, 2*C_f)
        #N - no. graphs (group of hits batched by some metric), P - no. hits per graph, K = no. nearest neighbours per hit, 2*C_f = feature dimension (includes original features and feature differences) i.e. no. features describing each hit
        # creates edge features by i) storing central hit fts, storing relative differences between central and neighboring hits

        # Edge MLP - Defines multi-layer perceptron to process edge features
        x = edge_features
        #sets x = edge_features computed in prev function

        for idx, channel in enumerate(channels):
            #Loops over channels list, specifies number of filters for each MLP layer
            #idx keeps track of the layer index
            x = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=1, 
                                    data_format='channels_last', 
                                    use_bias=not with_bn,  # Ensuring BN compatibility, No bias if BN is applied
                                    kernel_initializer='HeNormal', 
                                    name=f"{name}_conv{idx}")(x)
            #Uses 1x1 convolutions to apply transformations independently to each edge
            #channel defines the number of filters (output feature dimensions)
            #(1,1) kernel ensures per-edge transformation without affecting spatial structure.
            #activation=activation applies a non-linearity (default = ReLU)
            #name=f'{name}_conv{idx}' assigns a unique name to each layer (easier to debug)

            if with_bn:
                x = keras.layers.BatchNormalization(name=f"{name}_bn{idx}")(x)
            #Batch normalization (with_bn) stabilizes training by reducing internal covariate shift - if enabled, runs here
            #Normalizes the output of the convolutional layer.

            if activation:
                x = keras.layers.Activation(activation, name=f"{name}_act{idx}")(x)


        """# shortcut
        sc = keras.layers.Conv2D(channels[-1], kernel_size=(1, 1), strides=1, data_format='channels_last',
                                 use_bias=False if with_bn else True, kernel_initializer='HeNormal', name='%s_sc_conv' % name)(tf.expand_dims(features, axis=2))
        if with_bn:
            sc = keras.layers.BatchNormalization(name='%s_sc_bn' % name)(sc)
        sc = tf.squeeze(sc, axis=2)"""

        # Final classification layer (1 output per edge)
        edge_logits = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name=f'{name}_output')(x)

        return edge_logits  # (N, P, K, 1)
        




# In[4]:


def get_edgeconv(input_shapes):
    """
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
    """

    points = keras.Input(name='points', shape=input_shapes['points'])
    features = keras.Input(name='features', shape=input_shapes['features']) if 'features' in input_shapes else None
    mask = keras.Input(name='mask', shape=input_shapes['mask']) if 'mask' in input_shapes else None

    num_points = tf.shape(points)[0]  # Dynamically get number of nodes in batch (num_points = batch size)

    points = tf.reshape(points, (-1, num_points, 6))  # Ensure correct format (batch_size, num_points, 6)
    features = tf.reshape(features, (-1, num_points, 6))  # Ensure batch size is included (batch_size, num_points, 6/C_f)
    K = 10  # Set a default value
    channels = [64, 128, 256]  # Define layer sizes

    edge_logits = edge_conv(points, features, num_points, K, channels, name='edgeconv')


    # New Model: Outputs edge classification logits directly
    GCNN_model = keras.Model(inputs=[points, features, mask], outputs=edge_logits, name='EdgeClassifierGCNN')

    return GCNN_model





# ### Load Dataset
# Change path to your train_dataset ( train + validation )

# In[ ]:

train_dataset = Dataset('ProcessedData/signal1_96_32652/train_data/train_data.parquet', data_format='channel_last')

"""print("Feature keys:", train_dataset.X.keys())  # Check feature names
for key in train_dataset.X.keys():
    print(f"First 5 samples from {key}:")
    print(train_dataset.X[key][:5])  # Print first 5 entries"""


# In[ ]:


GCNN_model_name = 'GCNN_model_test'        #set your GCNN_model (file) name
num_classes = 2                    #our task is binary classification so we only have two classes
input_shapes = {k:np.shape(train_dataset[k])[1:] for k in train_dataset.X}
GCNN_model = get_edgeconv(input_shapes)


# ### Learning Rate Strategy
# We have two learning rate strategy: <b>constant learning rate</b> and <b>learning rate decay</b>.
# 
# Also set up your <b>batch size</b> here.
# 
# If you want to use <b>learning rate decay</b> strategy, don't forget to enter your number of training epochs and validation split ratio here.

# In[ ]:


num_epochs = 10
batch_size = 64
validation_split=0.25
num_train_steps = (len(train_dataset['points'])*(1-validation_split) // batch_size) * num_epochs

frames_per_batch = 10  # You can change this value for different experiments

# Extract unique frame numbers
frames = sorted(set(train_dataset["framenumber"]))

# Create dictionaries mapping frame numbers to data
hits_dict = {frame: train_dataset["points"][train_dataset["framenumber"] == frame] for frame in frames}
labels_dict = {frame: train_dataset["y"][train_dataset["framenumber"] == frame] for frame in frames}

# Create batched dataset
batched_dataset = Batching(frames, hits_dict, labels_dict, frames_per_batch)
print(batched_dataset)


#Polynomial lr Decay
def lr_schedule(initial_learning_rate,end_learning_rate):
    lr = keras.optimizers.schedules.PolynomialDecay(
       initial_learning_rate = initial_learning_rate,
       end_learning_rate = end_learning_rate,
       decay_steps = num_train_steps)
    return lr

#If you want to use constant lr, activate this function and deactivate the above one
'''def lr_schedule(lr):
    lr = lr
    return lr'''


# ### Set up optimizer and Learning rate here

# In[ ]:


GCNN_model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0.01,0.001)),    #you can change optimizer and lr here
              metrics=['accuracy'])
GCNN_model.summary()


# In[12]:


checkpoint = keras.callbacks.ModelCheckpoint(filepath=f'ProcessedData/model_record/{GCNN_model_name}',           #filepath
                             monitor='val_accuracy',
                             verbose=10,
                             save_best_only=True)         
progress_bar = keras.callbacks.ProgbarLogger()
callbacks = [checkpoint, progress_bar]


# ### Shuffle training train_dataset

# In[13]:


train_dataset.shuffle()


# ### Train the GCNN_model
# Set <b>number of training epochs</b> and set <b>ratio</b> to split Dataset for training set and testing set. Start training.

# In[ ]:

# Train model
history = GCNN_model.fit(train_dataset.X, train_dataset.y,
          batch_size=batch_size,
          epochs=10,                                                    # --- set number of training epochs ---      
          validation_split=0.25,                                        # --- set ratio to split Dataset for training set and testing set ---
          shuffle=True,
          callbacks=callbacks)

GCNN_model.save(f"GCNN_model_record/{GCNN_model_name}")


# ### Prediction
# Set <b>testing train_dataset file path</b> here. Also we will save prediction scores which use to make plot.

# In[ ]:


import pickle

# Save training metrics
metrics = dict(history.history)
with open(f"GCNN_model_record/{GCNN_model_name}/GCNN_model_metrics.pickle", 'wb') as handle:
        pickle.dump(metrics , handle, protocol=4)
print(f"metrics saved: {GCNN_model_name}")


# In[ ]:


# Predict on test dataset
# Change path to testing dataset here
test_dataset = Dataset('ProcessedData/signal1_96_32652/test_data/test_data.parquet', data_format='channel_last')


# In[24]:


# Save to numpy files to use in make_plots2.py
scores = np.array(GCNN_model.predict(test_dataset.X)).astype(np.float64)
np.save(f'GCNN_model_record/{GCNN_model_name}/prediction_scores.npy', scores)
np.save('inputs/test_labels.npy', test_dataset.y)


# ### Simply visualize result

plt.hist(scores)
# %%
