import tensorflow as tf
import keras

# Define Hello World in Neural Network | Layer density has only 1
model = keras.Sequential([keras.layer.Dense(units=1, input_shape=[1])])

# optimizer and loss will improve the mathematical function guess on machine
# math is important, preferably learning it too, 
# but for now all the math function is already in package method/function
model.compile(optimizer='sgd', loss='mean_squared_error')
# loss function will measure the guess/bad guesses, and then give the data to the optimizer which figures out the next guesses
# then the logic itself, each guess should be better than the one before