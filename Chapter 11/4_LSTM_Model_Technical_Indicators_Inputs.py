import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from master_function import mass_import, rsi, ma, calculate_accuracy
from master_function import plot_train_test_values, multiple_data_preprocessing
from sklearn.metrics import mean_squared_error
from master_function import add_column, delete_column

# Calling the function and preprocessing the data
data = mass_import(0, 'W1')[:, -1]
data = rsi(np.reshape(data, (-1, 1)), 5, 0, 1)
data = ma(data, 5, 0, 2)
data[:, 2] = data[:, 0] - data[:, 2]
data = add_column(data, 1)
for i in range(len(data)):
    data[i, 3] = data[i, 0] - data[i - 1, 0]
data[:, 0] = data[:, -1]
data = delete_column(data, 3, 1)

# Setting the hyperparameters
num_lags = 6
train_test_split = 0.80
neurons = 500
num_epochs = 500
batch_size = 200

x_train, y_train, x_test, y_test = multiple_data_preprocessing(data, train_test_split)

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

# Adding a fifth layer
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