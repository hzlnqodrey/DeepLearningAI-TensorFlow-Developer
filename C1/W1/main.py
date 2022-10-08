import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

# Define Hello World in Neural Network | Layer density has only 1
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# optimizer and loss will improve the mathematical function guess on machine
# math is important, preferably learning it too, 
# but for now all the math function is already in package method/function
model.compile(optimizer='sgd', loss='mean_squared_error')
# loss function will measure the guess/bad guesses, and then give the data to the optimizer which figures out the next guesses
# then the logic itself, each guess should be better than the one before

# Set the data
x_value = [x for x in np.arange(-100.0, 1000000.0, 1.0)]
y_value = [y for y in np.arange(-100.0, 2000100.0, 2.0)]
xs = np.array(x_value, dtype=float)
ys = np.array(y_value, dtype=float)

# the training will takes place in the fit command
model.fit(xs, ys, epochs=500)
# epochs equals 500 value means that it will go through the training loop 500 times

# when the model has finished training
# it will give back values using the predict method
print(model.predict([10.0]))

# check tensorflow version
print(tf.__version__) 