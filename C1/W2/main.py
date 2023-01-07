import tensorflow as tf
from tensorflow import keras

# Load data from mnist database in keras

fashion_mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Neural Network Definition of 3 Layers

model = keras.Sequential([
    # So the first layer is a flatten layer with input shaping 28 by 28 | because we specify that input data (mnist data) is 28x28 array bit | we should expect the data to be in
    # the flatten takes this 28 by 28 square and turn it into a simple linear array
    keras.layers.Flatten(input_shape=(28, 28)),
    # the second layer is called hidden layer | 128 layers of NN 
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])