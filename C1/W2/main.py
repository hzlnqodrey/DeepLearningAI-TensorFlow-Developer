import tensorflow as tf
from tensorflow import keras

# Load data from mnist database in keras

fashion_mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()