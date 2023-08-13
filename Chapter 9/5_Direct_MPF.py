# Import the required libraries
from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from master_function import direct_mpf
from master_function import calculate_directional_accuracy
from sklearn.metrics import mean_squared_error

# Import the data
data = np.reshape(np.array(pd.read_excel('ISM_PMI.xlsx').dropna()), (-1))

# Setting the hyperparameters
num_lags = 10
train_test_split = 0.80
num_neurons_in_hidden_layers = 200
num_epochs = 200
batch_size = 10
forecast_horizon = 18

x_train, y_train, x_test, y_test = direct_mpf(data, num_lags, train_test_split, forecast_horizon)

# Designing the architecture of the model
model = Sequential()

# First hidden layer
model.add(Dense(num_neurons_in_hidden_layers, input_dim = num_lags, activation = 'relu'))  

# Second hidden layer
model.add(Dense(num_neurons_in_hidden_layers, activation = 'relu'))  

# Output layer
model.add(Dense(forecast_horizon))

# Compiling
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# Fitting (training) the model
model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch_size)

# Make predictions
y_predicted = model.predict(x_test)

# Plotting
plt.plot(y_predicted[-1], label = 'Predicted data', color = 'red', linewidth = 1)
plt.plot(y_test[-1], label = 'Test data', color = 'black', linestyle = 'dashed', linewidth = 2)
plt.grid()
plt.legend()

# Performance evaluation
y_test = y_test[-1]
y_predicted = y_predicted[-1]

print('---')
print('Directional Accuracy Test = ', round(calculate_directional_accuracy(y_predicted, y_test), 2), '%')
print('RMSE Test = ', round(np.sqrt(mean_squared_error(y_predicted, y_test)), 10))
print('Correlation Out-of-Sample Predicted/Test = ', round(np.corrcoef(np.reshape(y_predicted, (-1)), np.reshape(y_test, (-1)))[0][1], 3))
print('---')