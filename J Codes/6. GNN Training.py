from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import awkward as ak
import awkward0
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import keras


# ### k-nearest neighbors
# construct graph data using knn algorithm

# In[2]:

#Calculates euclidean distances in graph space between nodes, knn will use this output to determine
#what counts as being nn. Should be able to stay as is, general enough.
def batch_distance_matrix_general(A, B):
    with tf.name_scope('dmat'):
        r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
        r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
        m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
        D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
        return D

#collects the knn based on results from above? Or based on something else? I don't see D in here anywhere
def knn(num_points, k, topk_indices, features):
    # topk_indices: (N, P, K)
    # features: (N, P, C)
    with tf.name_scope('knn'):
        queries_shape = tf.shape(features)
        batch_size = queries_shape[0]
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(topk_indices, axis=3)], axis=3)  # (N, P, K, 2)
        return tf.gather_nd(features, indices)


# ### Edge Convulution operation
# Attention: use (1,1) kernel Conv2D to perform MLP

# In[3]:

#main message passing operation of GNN can make major theory tweaks here
def edge_conv(points, features, num_points, K, channels, with_bn=True, activation='relu', name='edgeconv'):
    """Modified EdgeConv for Edge Classification
    
    Args:
        points: (N, P, C_p) - Hit positions (N events, P hits, C_p position features)
        features: (N, P, C_f) - Hit features (N events, P hits, C_f feature channels)
        num_points: Number of hits per event
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
        D = batch_distance_matrix_general(points, points)  # (N, P, P), Pairwise distances between nodes in graph space?
        _, indices = tf.nn.top_k(-D, k=K + 1)  # (N, P, K+1), collects K+1 nearest neighbors of each node (including self)
        indices = indices[:, :, 1:]  # (N, P, K) removes the self-connection so that only the actual K-nearest neighbors remain

        # Get neighbor features
        knn_features = knn(num_points, K, indices, features)  # (N, P, K, C_f) 
        #extracts features of the K nearest neighbors for each hit and stores by calling knn function
        knn_features_center = tf.tile(tf.expand_dims(features, axis=2), (1, 1, K, 1))  # (N, P, K, C_f)
        #duplicates the central hit’s features (i.e., hit focusing on currently) so can compare with neighbors
        edge_features = tf.concat([knn_features_center, knn_features, tf.subtract(knn_features_center)], axis=-1)  # (N, P, K, 2*C_f)
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
        edge_embds = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name=f'{name}_output')(x)

        return edge_embds  # (N, P, K, 1)
        

# In[4]:


def get_edgeconv(num_classes, input_shapes):
    """
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
    """

    points = keras.Input(name='points', shape=input_shapes['points'])
    features = keras.Input(name='features', shape=input_shapes['features']) if 'features' in input_shapes else None
    mask = keras.Input(name='mask', shape=input_shapes['mask']) if 'mask' in input_shapes else None


    edge_logits = edge_conv(points, features, mask, name='edgeconv')

    # New Model: Outputs edge classification logits directly
    GCNN_model = keras.Model(inputs=[points, features, mask], outputs=edge_logits, name='EdgeClassifierGCNN')

    return GCNN_model


# #### Stack and pad arrays

# In[6]:


import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def stack_arrays(a, keys, axis=-1):
    flat_arr = np.stack([a[k].flatten() for k in keys], axis=axis)
    return ak.JaggedArray.fromcounts(a[keys[0]].counts, flat_arr)

def pad_array(a, maxlen, value=0., dtype='float64'):
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = np.array(s[:maxlen]).astype(dtype)
        x[idx, :len(trunc)] = trunc
    return x


# ### Adjust train_dataset

# In[7]:

class Dataset(object):

    def __init__(self, filepath, feature_dict = {}, label='tid_array', pad_len=100, data_format='channel_first'):
        self.filepath = filepath
        self.feature_dict = feature_dict
        if len(feature_dict)==0:
            feature_dict['points'] = ['pixelx_array', 'pixely_array', 'layer_array', 'station_array', 'ladder_array', 'chip_array']
            feature_dict['features'] = ['pixelx_array', 'pixely_array', 'layer_array', 'station_array', 'ladder_array', 'chip_array']
            feature_dict['mask'] = ['']
            #"pixelx_array": b,"pixely_array": c,"layer_array"

        self.label = label
        self.pad_len = pad_len
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format=='channel_first' else -1
        self._values = {}
        self._label = None
        self._load()

    def _load(self):
        logging.info('Start loading file %s' % self.filepath)
        counts = None

        a = ak.from_parquet(self.filepath)
        print(a)
            
        self._label = a[self.label]
            
        for k in self.feature_dict:
            cols = self.feature_dict[k]
            if not isinstance(cols, (list, tuple)):
                cols = [cols]
            arrs = []

            for col in cols:
                if counts is None:
                    counts = ak.count(a[col],axis=None)
                else:
                    assert np.array_equal(counts, ak.count(a[col],axis=None))
                arrs.append(pad_array(a[col], self.pad_len))
            self._values[k] = np.stack(arrs, axis=self.stack_axis)

        logging.info('Finished loading file %s' % self.filepath)


    def __len__(self):
        return len(self._label)

    def __getitem__(self, key):
        if key==self.label:
            return self._label
        else:
            return self._values[key]
    
    @property
    def X(self):
        return self._values
    
    @property
    def y(self):
        return self._label

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        shuffle_indices = np.arange(self.__len__())
        np.random.shuffle(shuffle_indices)
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]


# ### Load Dataset
# Change path to your train_dataset ( train + validation )

# In[ ]:


train_dataset = Dataset('ProcessedData/signal1_96_32652/train_data/train_data.parquet', data_format='channel_last')


# In[ ]:


GCNN_model_name = 'GCNN_model_test'        #set your GCNN_model (file) name
num_classes = 2                    #our task is binary classification so we only have two classes
input_shapes = {k:np.shape(train_dataset[k])[1:] for k in train_dataset.X}
GCNN_model = get_edgeconv(num_classes, input_shapes)


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

#Polynomial lr Decay
def lr_schedule(initial_learning_rate,end_learning_rate):
    lr = tf.keras.optimizers.schedules.PolynomialDecay(
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
