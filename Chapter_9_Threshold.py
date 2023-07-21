import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
# from master_function_dlf import *

# Setting the hyperparameters
num_lags = 500
train_test_split = 0.80 
num_neurons_in_hidden_layers = 256 
num_epochs = 100 
batch_size = 10
threshold = 0.0015

# Fetch the historical price data
raw_data = mass_import(1, 'D1')
raw_data = raw_data[:, 3]

# creating the lagged values array to forecast the returns from previous returns
my_data = np.zeros((raw_data.shape[0], num_lags))

for lag in range(num_lags):
    my_data[:, lag] = np.roll(raw_data, lag + 1)
    
my_data = my_data[num_lags:, ]

# Taking the returns values of every time step
my_data = np.diff(my_data, axis = 0) / my_data[:-1]

# Splitting the data 80/20
split = int(train_test_split * len(my_data))

# Defining the training set as the first 80% of the independent variables
x_train = my_data[:split, 1:num_lags]

# Defining the training set as the first 80% of the dependent variables
y_train = my_data[:split, 0]
y_train = np.reshape(y_train, (-1, 1))

# Defining the test set as the last 20% of the independent variables
x_test = my_data[split:, 1:num_lags]

# Defining the test set as the last 20% of the dependent variables
y_test = my_data[split:, 0]
y_test = np.reshape(y_test, (-1, 1))

# Designing the architecture of the model
model = Sequential()

# First hidden layer
model.add(Dense(num_neurons_in_hidden_layers, input_dim = num_lags - 1, activation = 'relu'))  

# Second hidden layer
model.add(Dense(num_neurons_in_hidden_layers, activation = 'relu'))  

# Output layer
model.add(Dense(1))

# Compiling
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch_size)

y_predicted = model.predict(x_test)

def threshold_forecasting(predictions, threshold):
    
    for i in range(len(predictions)):
        
        if predictions[i] > threshold:
            
            predictions[i] = predictions[i]

        elif predictions[i] < -threshold:
            
            predictions[i] = predictions[i]
            
        else:
            
            predictions[i] = 0
                
    return predictions

y_predicted = threshold_forecasting(y_predicted, threshold)

# Plotting
prediction_window = 50
first = 25
second = 25
plotting_time_series = np.zeros((prediction_window, 3))
plotting_time_series[0:first, 0] = y_train[-first:, 0]
plotting_time_series[first:, 1] = y_test[0:second, 0]
plotting_time_series[first:, 2] = y_predicted[0:second, 0]
 
plotting_time_series[0:first, 1] = plotting_time_series[0:first, 1] / 0
plotting_time_series[0:first, 2] = plotting_time_series[0:first, 2] / 0
plotting_time_series[first:, 0] = plotting_time_series[first:, 0] / 0
 
plt.plot(plotting_time_series[:, 0], label = 'Training data', color = 'black', linewidth = 2.5)
plt.plot(plotting_time_series[:, 1], label = 'Test data', color = 'black', linestyle = 'dashed', linewidth = 2)
plt.plot(plotting_time_series[:, 2], label = 'Predicted data', color = 'red', linewidth = 1)
plt.axvline(x = first, color = 'black', linestyle = '--', linewidth = 1)
plt.grid()
plt.legend()
