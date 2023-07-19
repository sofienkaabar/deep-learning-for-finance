# Import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Import the data
raw_data = pd.read_excel('ISM_PMI.xlsx')
raw_data = raw_data.dropna()
raw_data = np.reshape(np.array(raw_data), (-1))

'''
plt.plot(raw_data[-905:], label = 'ISM PMI')
plt.legend()
plt.grid()
'''

# Setting the hyperparameters
num_lags = 40
train_test_split = 0.8
num_neurons_in_hidden_layers = 100
num_epochs = 200
batch_size = 12

# Creating the lagged values array
my_data = np.zeros((raw_data.shape[0], num_lags))

for lag in range(num_lags):
    my_data[:, lag] = np.roll(raw_data, lag + 1)
    
my_data = my_data[num_lags:, ]

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

# Third hidden layer
model.add(Dense(num_neurons_in_hidden_layers, activation = 'relu'))  

# Fourth hidden layer
model.add(Dense(num_neurons_in_hidden_layers, activation = 'relu')) 

# Output layer
model.add(Dense(1))

# Compiling
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch_size)

# Training set evaluation
in_sample_predictions = model.predict(x_train)

'''
plt.plot(in_sample_predictions[-200:, ], label = 'Predicted', linestyle = 'dashed', color = 'red')
plt.plot(y_train[-200:, ], label = 'Real', color = 'black')
plt.grid()
plt.legend()
'''

# Latest values to use as inputs
x_test = x_test[-1]
x_test = np.reshape(x_test, (-1, 1))
x_test = np.transpose(x_test)

y_predicted = []

for i in range(len(y_test)):
    
    # Predict over the last x_test values
    predicted_value = model.predict(x_test)
    y_predicted = np.append(y_predicted, predicted_value)
    
    # Re-inserting the latest prediction into x_test array
    x_test = np.transpose(x_test)
    x_test = np.append(x_test, predicted_value)
    x_test = x_test[1:, ]
    x_test = np.reshape(x_test, (-1, 1))
    x_test = np.transpose(x_test)

# Plotting
prediction_window = 200
first = 60
second = 140
plotting_time_series = np.zeros((prediction_window, 3))
plotting_time_series[0:first, 0] = y_train[-first:, 0]
plotting_time_series[first:, 1] = y_test[0:second, 0]
plotting_time_series[first:, 2] = y_predicted[0:second]
 
plotting_time_series[0:first, 1] = plotting_time_series[0:first, 1] / 0
plotting_time_series[0:first, 2] = plotting_time_series[0:first, 2] / 0
plotting_time_series[first:, 0] = plotting_time_series[first:, 0] / 0
 
plt.plot(plotting_time_series[:, 0], label = 'Training data', color = 'black', linewidth = 2.5)
plt.plot(plotting_time_series[:, 1], label = 'Test data', color = 'black', linestyle = 'dashed', linewidth = 2)
plt.plot(plotting_time_series[:, 2], label = 'Predicted data', color = 'red', linewidth = 1)
plt.axvline(x = first, color = 'black', linestyle = '--', linewidth = 1)
plt.grid()
plt.legend()

'''
plt.plot(y_test)
plt.plot(y_predicted)
'''
