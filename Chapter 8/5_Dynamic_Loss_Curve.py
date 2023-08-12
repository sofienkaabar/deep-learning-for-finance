# Importing libraries
from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from master_function import data_preprocessing, plot_train_test_values
from master_function import calculate_accuracy, model_bias
from sklearn.metrics import mean_squared_error

# Set the start and end dates for the data
start_date = '1990-01-01'
end_date   = '2023-06-01'

# Fetch S&P 500 price data
data = np.array((pdr.get_data_fred('SP500', start = start_date, end = end_date)).dropna())

# Difference the data and make it stationary
data = np.diff(data[:, 0])

# Setting the hyperparameters
num_lags = 100
train_test_split = 0.80
num_neurons_in_hidden_layers = 200
num_epochs = 100
batch_size = 5

# Creating the training and test sets
x_train, y_train, x_test, y_test = data_preprocessing(data, num_lags, train_test_split)

# Designing the architecture of the model
model = Sequential()

# First hidden layer
model.add(Dense(num_neurons_in_hidden_layers, input_dim = num_lags, activation = 'relu'))  

# Second hidden layer
model.add(Dense(num_neurons_in_hidden_layers, activation = 'relu'))  

# Output layer
model.add(Dense(1))
    
# Compiling
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# Importing tensorflow (Make sure to pip install it)
import tensorflow as tf

# Preparing the storing arrays
losses = []
epochs = []

# Main function
class LossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        losses.append(logs['loss'])
        epochs.append(epoch + 1)
        plt.clf()
        plt.plot(epochs, losses, marker = 'o')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.grid(True)
        plt.pause(0.01)

# Fitting with a callback to chart the loss function
model.fit(x_train, np.reshape(y_train, (-1, 1)), epochs = 100, verbose = 0, callbacks = [LossCallback()])
plt.show()
