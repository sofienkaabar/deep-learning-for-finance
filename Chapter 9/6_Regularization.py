import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import pandas_datareader as pdr
from master_function import data_preprocessing, plot_train_test_values
from master_function import calculate_directional_accuracy
from sklearn.metrics import mean_squared_error

# Set the start and end dates for the data
start_date = '1990-01-01'
end_date   = '2023-06-01'

# Fetch S&P 500 price data
data = np.array((pdr.get_data_fred('SP500', start = start_date, end = end_date)).dropna())

# Calculating a rolling autocorrelation measure
rolling_autocorr = pd.DataFrame(data).rolling(window = 20).apply(lambda x: x.autocorr(lag = 1)).dropna()
rolling_autocorr = np.reshape(np.array(rolling_autocorr), (-1))

# Plotting
fig, axes = plt.subplots(nrows = 2, ncols = 1)

axes[0].plot(data[-350:,], label = 'S&P 500', linewidth = 1.5)
axes[1].plot(rolling_autocorr[-350:,], label = '20-Day Autocorrelation', color = 'orange', linewidth = 1.5)

axes[0].legend()
axes[1].legend()

axes[0].grid()
axes[1].grid()

axes[1].axhline(y = 0.95, color = 'black', linestyle = 'dashed') 

# Setting the hyperparameters
num_lags = 500 
train_test_split = 0.80 
num_neurons_in_hidden_layers = 128 
num_epochs = 100 
batch_size = 20

# Creating the training and test sets
x_train, y_train, x_test, y_test = data_preprocessing(rolling_autocorr, num_lags, train_test_split)

# Reshape the data for LSTM input
x_train = x_train.reshape((-1, num_lags, 1))
x_test = x_test.reshape((-1, num_lags, 1))

# Create the LSTM model
model = Sequential()
model.add(LSTM(units = num_neurons_in_hidden_layers, input_shape = (num_lags, 1)))

# Adding batch normalization and dropout
model.add(BatchNormalization())
model.add(Dropout(0.1)) 

# Adding the output layer
model.add(Dense(units = 1))

# Compile the model
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# Early stopping implementation
early_stopping = EarlyStopping(monitor = 'loss', patience = 15, restore_best_weights = True)

# Train the model
model.fit(x_train, y_train, epochs = num_epochs , batch_size = batch_size, callbacks = [early_stopping])

# Predicting in-sample
y_predicted_train = np.reshape(model.predict(x_train), (-1, 1))

# Predicting out-of-sample
y_predicted = np.reshape(model.predict(x_test), (-1, 1))

# plotting
plot_train_test_values(300, 50, y_train, y_test, y_predicted)

# Performance evaluation
print('---')
print('Accuracy Train = ', round(calculate_directional_accuracy(y_predicted_train, y_train), 2), '%')
print('Accuracy Test = ', round(calculate_directional_accuracy(y_predicted, y_test), 2), '%')
print('RMSE Train = ', round(np.sqrt(mean_squared_error(y_predicted_train, y_train)), 10))
print('RMSE Test = ', round(np.sqrt(mean_squared_error(y_predicted, y_test)), 10))
print('Correlation In-Sample Predicted/Train = ', round(np.corrcoef(np.reshape(y_predicted_train, (-1)), y_train)[0][1], 3))
print('Correlation Out-of-Sample Predicted/Test = ', round(np.corrcoef(np.reshape(y_predicted, (-1)), np.reshape(y_test, (-1)))[0][1], 3))
print('---')
