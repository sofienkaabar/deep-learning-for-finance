# importing libraries
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# importing the dataset since 2014
raw_data = mass_import(0, 'H1')
raw_data = raw_data[:, 3]

# creating the lagged values array to forecast the returns from previous returns
num_lags = 6
my_data = np.zeros((raw_data.shape[0], num_lags))

for lag in range(num_lags):
    my_data[:, lag] = np.roll(raw_data, lag + 1)
    
my_data = my_data[6:, ]

# taking the returns values of every time step
my_data = np.diff(my_data, axis = 0) / my_data[:-1]

# splitting the data 80/20
train_test_split = 0.80
split = int(train_test_split * len(my_data))

# defining the training set as the first 80% of the independent variables
x_train = my_data[:split, 1:6]

# defining the training set as the first 80% of the dependent variables
y_train = my_data[:split, 0]
y_train = np.reshape(y_train, (-1, 1))

# defining the test set as the last 20% of the independent variables
x_test = my_data[split:, 1:6]

# defining the test set as the last 20% of the dependent variables
y_test = my_data[split:, 0]
y_test = np.reshape(y_test, (-1, 1))

# Fitting the model
regressor = make_pipeline(StandardScaler(), SVR(kernel = 'rbf', C = 1, gamma = 0.04, epsilon = 0.01))
regressor.fit(x_train, y_train)

# Predicting 
y_predicted = regressor.predict(x_test)
y_predicted = np.reshape(y_predicted, (-1, 1))

# plotting
plotting_time_series = np.zeros((100, 3))
plotting_time_series[0:70, 0] = y_train[-70:, 0]
plotting_time_series[70:, 1] = y_test[0:30, 0]
plotting_time_series[70:, 2] = y_predicted[0:30, 0]
 
plotting_time_series[0:70, 1] = plotting_time_series[0:70, 1] / 0
plotting_time_series[0:70, 2] = plotting_time_series[0:70, 2] / 0
plotting_time_series[70:, 0] = plotting_time_series[70:, 0] / 0
 
plt.plot(plotting_time_series[:, 0], label = 'Training data', color = 'black', linewidth = 2.5)
plt.plot(plotting_time_series[:, 1], label = 'Test data', color = 'black', linestyle = 'dashed', linewidth = 2)
plt.plot(plotting_time_series[:, 2], label = 'Predicted data', color = 'red', linewidth = 1)
plt.axvline(x = 70, color = 'black', linestyle = '--', linewidth = 1)
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



