import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from master_function import data_preprocessing
from master_function import plot_train_test_values, calculate_accuracy
from sklearn.metrics import mean_squared_error
from master_function import add_column, delete_row, volatility

# Calling the function and preprocessing the data
data = pd.read_excel('Bitcoin_Daily_Historical_Data.xlsx').values
data = volatility(data, 10, 0, 1)
data = data[:, -1]

# Differencing for stationarity
data = np.diff(data)

# Checking for stationarity
from statsmodels.tsa.stattools import adfuller
print('p-value: %f' % adfuller(data)[1])

# Setting the hyperparameters
num_lags = 300
train_test_split = 0.80
neurons = 80
num_epochs = 100
batch_size = 500

# Creating the training and test sets
x_train, y_train, x_test, y_test = data_preprocessing(data, num_lags, train_test_split)

# Reshape the data to 3D for LSTM input
x_train = x_train.reshape((-1, num_lags, 1))
x_test = x_test.reshape((-1, num_lags, 1))

# Create the LSTM model
model = Sequential()

# Adding a first layer
model.add(LSTM(units = neurons, input_shape = (num_lags, 1)))

# Adding a second layer
model.add(Dense(neurons, activation = 'relu')) 

# Adding a third layer
model.add(Dense(neurons, activation = 'relu')) 

# Adding a fourth layer
model.add(Dense(neurons, activation = 'relu')) 

# Adding the output layer 
model.add(Dense(units = 1))

# Compiling the model
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# Fitting the model
model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch_size)

# Predicting in the training set for illustrative purposes
y_predicted_train = model.predict(x_train)

# Predicting in the test set
y_predicted = model.predict(x_test)

# Plotting
plot_train_test_values(100, 50, y_train, y_test, y_predicted)

# Performance evaluation
print('---')
print('Accuracy Train = ', round(calculate_accuracy(y_predicted_train, y_train), 2), '%')
print('Accuracy Test = ', round(calculate_accuracy(y_predicted, y_test), 2), '%')
print('RMSE Train = ', round(np.sqrt(mean_squared_error(y_predicted_train, y_train)), 10))
print('RMSE Test = ', round(np.sqrt(mean_squared_error(y_predicted, y_test)), 10))
print('Correlation In-Sample Predicted/Train = ', round(np.corrcoef(np.reshape(y_predicted_train, (-1)), y_train)[0][1], 3))
print('Correlation Out-of-Sample Predicted/Test = ', round(np.corrcoef(np.reshape(y_predicted, (-1)), np.reshape(y_test, (-1)))[0][1], 3))
print('---')