import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Is tf.keras.layers available?", hasattr(tf.keras, "layers"))

# Try to import a specific layer
try:
    from tensorflow.keras.layers import BatchNormalization
    print("BatchNormalization imported successfully!")
except ModuleNotFoundError as e:
    print("Error:", e)
