# importing libraries
from keras.models import Sequential
from keras.layers import Dense
import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from master_function import data_preprocessing
from master_function import plot_train_test_values, calculate_accuracy, model_bias
from sklearn.metrics import mean_squared_error

# Generate the sine wave
x = np.arange(0,10000)
data = np.diff((np.sin(x) + 1) * 10 + 2 * x + np.random.rand(10000) * 5)

# Creating the lagged values array to forecast the returns from previous returns
num_lags = 500
train_test_split = 0.80

# Creating the training and test sets
x_train, y_train, x_test, y_test = data_preprocessing(data, num_lags, train_test_split)

# Designing the architecture of the model
model = Sequential()

# First hidden layer
model.add(Dense(24, input_dim = num_lags, activation = 'relu'))  

# Second hidden layer
model.add(Dense(24, activation = 'relu'))  

# Output layer
model.add(Dense(1))

# Compiling
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

model.fit(x_train, y_train, epochs = 100, batch_size = 32)

# Predicting in-sample
y_predicted_train = np.reshape(model.predict(x_train), (-1, 1))

# Predicting out-of-sample
y_predicted = np.reshape(model.predict(x_test), (-1, 1))

# plotting
plot_train_test_values(100, 50, y_train, y_test, y_predicted)

# Performance evaluation
print('---')
print('Accuracy Train = ', round(calculate_accuracy(y_predicted_train, y_train), 2), '%')
print('Accuracy Test = ', round(calculate_accuracy(y_predicted, y_test), 2), '%')
print('RMSE Train = ', round(np.sqrt(mean_squared_error(y_predicted_train, y_train)), 10))
print('RMSE Test = ', round(np.sqrt(mean_squared_error(y_predicted, y_test)), 10))
print('Correlation In-Sample Predicted/Train = ', round(np.corrcoef(np.reshape(y_predicted_train, (-1)), y_train)[0][1], 3))
print('Correlation Out-of-Sample Predicted/Test = ', round(np.corrcoef(np.reshape(y_predicted, (-1)), np.reshape(y_test, (-1)))[0][1], 3))
print('Model Bias = ', round(model_bias(y_predicted), 2))
print('---')