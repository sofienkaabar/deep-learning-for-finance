# Import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Import the data
raw_data = pd.read_excel('Temperature_Basel.xlsx')
raw_data = raw_data.dropna()
raw_data = np.reshape(np.array(raw_data), (-1))

# Prepare the multi-output architecture
def prepare_data(data, time_steps, forecast_horizon):
    x, y = [], []
    for i in range(len(data) - time_steps - forecast_horizon + 1):
        x.append(data[i:i + time_steps])
        y.append(data[i + time_steps:i + time_steps + forecast_horizon])
    return np.array(x), np.array(y)

# Setting the hyperparameters
num_lags = 500
train_test_split = 0.80
num_neurons_in_hidden_layers = 128
num_epochs = 50
batch_size = 12
forecast_horizon = 500

# Prepare the arrays
x, y = prepare_data(raw_data, num_lags, forecast_horizon)

# Splitting the data 80/20
split = int(train_test_split * len(raw_data))

x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

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