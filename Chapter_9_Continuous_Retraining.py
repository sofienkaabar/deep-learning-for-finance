import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# from master_function_dlf import *

# Setting the hyperparameters
num_lags = 10
train_test_split = 0.80 

# Fetch the historical price data
raw_data = mass_import(1, 'W1')
raw_data = raw_data[:, 3]

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

model = LinearRegression()

model.fit(x_train, y_train)

# Store the new forecasts
y_predicted = []

# Reshape x_test to forecast one period
latest_values = x_test[0]
latest_values = np.reshape(latest_values, (-1, 1))
latest_values = np.transpose(latest_values)

# Isolate the real values for comparison
y_test_store = y_test

for i in range(len(y_test)):

    try:
        
        # Predict over the first x_test data
        predicted_value = model.predict(latest_values)
        
        # Store the prediction in an array
        y_predicted = np.append(y_predicted, predicted_value)
        
        # Adding the first test values to the last training values
        x_train = np.concatenate((x_train, latest_values), axis = 0)
        y_train = np.append(y_train, y_test[0])
        
        # Removing the first test values from the test arrays
        y_test = y_test[1:]
        x_test = x_test[1:, ]
        
        # Retraining
        model.fit(x_train, y_train)
        
        # Selecting the first values of the test set
        latest_values = x_test[0]
        latest_values = np.reshape(latest_values, (-1, 1))
        latest_values = np.transpose(latest_values)

    except IndexError:
        
        pass
    
plt.plot(y_predicted, label = 'Predicted data', color = 'red', linewidth = 1)

plt.grid()
plt.legend()

def accuracy(predicted_returns, real_returns):
    
    hits = sum((np.sign(predicted_returns) == np.sign(real_returns)))
    
    total_samples = len(predicted_returns)
    
    accuracy = hits / total_samples
    
    return accuracy * 100

print('Accuracy = ', round(accuracy(y_predicted, y_test_store)[0], 2), '%')


