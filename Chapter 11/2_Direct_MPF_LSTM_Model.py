import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from master_function import import_cot_data, direct_mpf
from master_function import calculate_directional_accuracy
from sklearn.metrics import mean_squared_error

# Calling the function and preprocessing the data
CAD = 'CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE'
data = import_cot_data(2010, 2023, CAD)
data = np.array(data.iloc[:, -1], dtype = np.float64)

# Setting the hyperparameters
num_lags = 100
train_test_split = 0.80
neurons = 400
num_epochs = 200
batch_size = 10
forecast_horizon = 100

# Prepare the arrays
x_train, y_train, x_test, y_test = direct_mpf(data, 
                                              num_lags, 
                                              train_test_split, 
                                              forecast_horizon)

# Reshape the data to 3D for LSTM input
x_train = x_train.reshape((-1, num_lags, 1))
x_test = x_test.reshape((-1, num_lags, 1))

# Create the LSTM model
model = Sequential()

# Adding a first layer
model.add(LSTM(units = neurons, input_shape = (num_lags, 1)))

# Adding a second layer
model.add(Dense(neurons, activation = 'relu')) 

# Adding the output layer 
model.add(Dense(units = forecast_horizon))

# Compiling the model
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# Fitting the model
model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch_size)

# Predicting in the test set
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