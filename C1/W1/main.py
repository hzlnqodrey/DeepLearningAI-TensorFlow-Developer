import tensorflow as tf
import keras
import numpy as np
import pandas as pd

# Define Hello World in Neural Network | Layer density has only 1
model = keras.Sequential([keras.layer.Dense(units=1, input_shape=[1])])

# optimizer and loss will improve the mathematical function guess on machine
# math is important, preferably learning it too, 
# but for now all the math function is already in package method/function
model.compile(optimizer='sgd', loss='mean_squared_error')
# loss function will measure the guess/bad guesses, and then give the data to the optimizer which figures out the next guesses
# then the logic itself, each guess should be better than the one before

# Set the data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
