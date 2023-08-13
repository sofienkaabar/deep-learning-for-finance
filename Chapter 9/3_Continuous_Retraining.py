import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from master_function import data_preprocessing, mass_import
from master_function import plot_train_test_values, calculate_accuracy, model_bias
from sklearn.metrics import mean_squared_error

# Importing the time series
data = np.diff(mass_import(0, 'D1')[:, 3])

# Setting the hyperparameters
num_lags = 15
train_test_split = 0.80 

# Creating the training and test sets
x_train, y_train, x_test, y_test = data_preprocessing(data, num_lags, train_test_split)

# Fitting the model
model = LinearRegression()
model.fit(x_train, y_train)

# Predicting in-sample
y_predicted_train = np.reshape(model.predict(x_train), (-1, 1))

# Store the new forecasts
y_predicted = []

# Reshape x_test to forecast one period
latest_values = np.transpose(np.reshape(x_test[0], (-1, 1)))

# Isolate the real values for comparison
y_test_store = y_test
y_train_store = y_train

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
        latest_values = np.transpose(np.reshape(x_test[0], (-1, 1)))
    except IndexError:
        pass
    
# plotting
plot_train_test_values(100, 50, y_train, y_test_store, y_predicted)

# Performance evaluation
print('---')
print('Accuracy Train = ', round(calculate_accuracy(y_predicted_train, y_train_store), 2), '%')
print('Accuracy Test = ', round(calculate_accuracy(y_predicted, y_test_store), 2), '%')
print('RMSE Train = ', round(np.sqrt(mean_squared_error(y_predicted_train, y_train_store)), 10))
print('RMSE Test = ', round(np.sqrt(mean_squared_error(y_predicted, y_test_store)), 10))
print('Correlation In-Sample Predicted/Train = ', round(np.corrcoef(np.reshape(y_predicted_train, (-1)), y_train_store)[0][1], 3))
print('Correlation Out-of-Sample Predicted/Test = ', round(np.corrcoef(np.reshape(y_predicted, (-1)), np.reshape(y_test_store, (-1)))[0][1], 3))
print('Model Bias = ', round(model_bias(y_predicted), 2))
print('---')

