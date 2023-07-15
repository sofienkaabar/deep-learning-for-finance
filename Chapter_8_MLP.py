# Importing libraries
from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr

# Setting the hyperparameters
num_lags = 100
train_test_split = 0.80
num_neurons_in_hidden_layers = 2
num_epochs = 500
batch_size = 2

# Set the start and end dates for the data
start_date = '1990-01-01'
end_date   = '2023-06-01'

# Fetch S&P 500 price data
raw_data = pdr.get_data_fred('SP500', start = start_date, end = end_date)
raw_data = raw_data.dropna()
raw_data = np.array(raw_data)
raw_data = np.reshape(raw_data, (-1))

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

# Plotting
prediction_window = 200
first = 100
second = 100
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

# Performance evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error

def accuracy(predicted_returns, real_returns):
    
    hits = sum((np.sign(predicted_returns) == np.sign(real_returns)))
    
    total_samples = len(predicted_returns)
    
    accuracy = hits / total_samples
    
    return accuracy * 100

def model_bias(predicted_returns):
    
    bullish_forecasts = np.sum(predicted_returns > 0)
    bearish_forecasts = np.sum(predicted_returns < 0)
    
    return bullish_forecasts / bearish_forecasts

print('Accuracy = ', round(accuracy(y_predicted, y_test)[0], 2), '%')
print('RMSE = ', round(np.sqrt(mean_squared_error(y_test, y_predicted)), 10))
print('MAE = ', round(mean_absolute_error(y_test, y_predicted), 10))
print('Model Bias = ', round(model_bias(y_predicted), 2))


