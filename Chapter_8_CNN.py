import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import pandas_datareader as pdr

# Setting the hyperparameters
num_lags = 100 
train_test_split = 0.80 
filters = 64 
kernel_size = 4
pool_size = 2
num_epochs = 100 
batch_size = 8

# Set the start and end dates for the data
start_date = '1990-01-01'
end_date   = '2023-06-01'

# Fetch S&P 500 price data
data = pdr.get_data_fred('SP500', start = start_date, end = end_date)
data = data.dropna()
data = np.array(data)
data = np.diff(data, axis = 0) / data[:-1]

# Prepare the data for training
sequences = []
targets = []
for i in range(len(data) - num_lags):
    sequences.append(data[i:i+num_lags])
    targets.append(data[i+num_lags])

# Convert the data to numpy arrays
sequences = np.array(sequences)
targets = np.array(targets)

# Split the data into training and testing sets
split_index = int(train_test_split * len(sequences))

x_train = sequences[:split_index]
y_train = targets[:split_index]

x_test = sequences[split_index:]
y_test = targets[split_index:]
y_test = np.reshape(y_test, (-1, 1))

# Reshape the data for 1D convolutional layer
x_train = x_train.reshape((-1, num_lags, 1))
x_test = x_test.reshape((-1, num_lags, 1))

# Create the temporal CNN model
model = Sequential()
model.add(Conv1D(filters = filters, kernel_size = kernel_size, activation = 'relu', 
                 input_shape = (num_lags, 1)))
model.add(MaxPooling1D(pool_size = pool_size))
model.add(Flatten())
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer = 'adam', loss = 'mse')

# Train the model
model.fit(x_train, y_train, epochs = num_epochs , batch_size = batch_size)

# Predict using the model
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

# performance evaluation
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