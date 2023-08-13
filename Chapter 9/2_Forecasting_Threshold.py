import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from master_function import data_preprocessing, mass_import
from master_function import plot_train_test_values, forecasting_threshold

# Setting the hyperparameters
num_lags = 500
train_test_split = 0.80 
num_neurons_in_hidden_layers = 256 
num_epochs = 100 
batch_size = 10
threshold = 0.0015

# Fetch the historical price data
data = np.diff(mass_import(0, 'D1')[:, 3])

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

# Fitting
model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch_size)

# Predicting
y_predicted = model.predict(x_test)

# Threshold function
y_predicted = forecasting_threshold(y_predicted, threshold)

# Plotting
plot_train_test_values(100, 50, y_train, y_test, y_predicted)
