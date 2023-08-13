# Importing libraries
from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from master_function import data_preprocessing, plot_train_test_values, recursive_mpf
from master_function import calculate_directional_accuracy
from sklearn.metrics import mean_squared_error

# Import the data
data = np.reshape(np.array(pd.read_excel('Temperature_Basel.xlsx').dropna()), (-1))

# Setting the hyperparameters
num_lags = 500
train_test_split = 0.8
num_neurons_in_hidden_layers = 100
num_epochs = 200
batch_size = 12

# Creating the training and test sets
x_train, y_train, x_test, y_test = data_preprocessing(data, num_lags, train_test_split)

# Designing the architecture of the model
model = Sequential()

# First hidden layer
model.add(Dense(num_neurons_in_hidden_layers, input_dim = num_lags, activation = 'relu'))  

# Second hidden layer
model.add(Dense(num_neurons_in_hidden_layers, activation = 'relu'))  

# Third hidden layer
model.add(Dense(num_neurons_in_hidden_layers, activation = 'relu'))  

# Fourth hidden layer
model.add(Dense(num_neurons_in_hidden_layers, activation = 'relu')) 

# Output layer
model.add(Dense(1))

# Compiling
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# Fitting the model
model.fit(x_train, np.reshape(y_train, (-1, 1)), epochs = num_epochs, batch_size = batch_size)

# Predicting in-sample
y_predicted_train = np.reshape(model.predict(x_train), (-1, 1))

# Predicting in the test set on a recursive basis
x_test, y_predicted = recursive_mpf(x_test, y_test, num_lags, model, architecture = 'MLP')

# Plotting
plot_train_test_values(1000, 50, y_train, y_test, y_predicted)

# Performance evaluation
print('---')
print('Directional Accuracy Test = ', round(calculate_directional_accuracy(y_predicted, y_test), 2), '%')
print('RMSE Test = ', round(np.sqrt(mean_squared_error(y_predicted, y_test)), 10))
print('Correlation Out-of-Sample Predicted/Test = ', round(np.corrcoef(np.reshape(y_predicted, (-1)), np.reshape(y_test, (-1)))[0][1], 3))
print('---')

