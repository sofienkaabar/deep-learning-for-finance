# Importing the required library
import pandas_datareader as pdr
from statsmodels.tsa.stattools import adfuller

# Setting the beginning and end of the historical data
start_date = '1950-01-01'
end_date   = '2023-01-23'

# Creating a dataframe and downloading the CPI data
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Printing the latest five observations of the dataframe
print(cpi.tail())

# Checking if there are nan values in the CPI dataframe
count_nan = cpi['CPIAUCSL'].isnull().sum()

# Printing the result
print('Number of nan values in the CPI dataframe: ' + str(count_nan))

# Transforming the CPI into a year-on-year measure
cpi = cpi.pct_change(periods = 12, axis = 0) * 100

# Dropping the nan values from the rows
cpi = cpi.dropna()

# Applying the ADF test on the CPI data
print('p-value: %f' % adfuller(cpi)[1])

# Creating a dataframe and downloading the CPI data
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Applying the ADF test on the CPI data
print('p-value: %f' % adfuller(cpi)[1])

# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt

# Creating the first time series using sine waves
length = np.pi * 2 * 5
sinewave = np.sin(np.arange(0, length, length / 1000))

# Creating the second time series using trending sine waves
sinewave_ascending = np.sin(np.arange(0, length, length / 1000))

# Defining the trend variable
a = 0.01

# Looping to add a trend factor
for i in range(len(sinewave_ascending)): 
    sinewave_ascending[i] = a + sinewave_ascending[i]
    a = 0.01 + a
    
# Plotting the series
plt.plot(sinewave, label = 'Sine Wave', color = 'black')
plt.plot(sinewave_ascending, label = 'Ascending Sine Wave', color = 'grey')

# Calling the grid function for better interpretability
plt.grid()

# Calling the legend function to show the labels
plt.legend()

# Showing the plot
plt.show()

# ADF testing | Normal sine wave
adfuller(sinewave) 
print('p-value: %f' % adfuller(sinewave)[1])

# ADF testing | Ascending sine wave
adfuller(sinewave_ascending) 
print('p-value: %f' % adfuller(sinewave_ascending)[1])

# Importing the KPSS library
from statsmodels.tsa.stattools import kpss

# KPSS testing | Normal sine wave
print('p-value: %f' % kpss(sinewave)[1])

# KPSS testing | Ascending sine wave
print('p-value: %f' % kpss(sinewave_ascending)[1])

# KPSS testing while taking into account the trend | Ascending sine wave
print('p-value: %f' % kpss(sinewave_ascending, regression = 'ct')[1])

'''
The 'ct' argument is used to check if the dataset is stationary 
around a trend. By default, the argument is 'c' which is is used
to check if the data is stationary around a constant.
'''

# Applying the KPSS (no trend consideration) test on the CPI data
print('p-value: %f' % kpss(cpi)[1])

# Applying the KPSS (with trend consideration) test on the CPI data
print('p-value: %f' % kpss(cpi, regression = 'ct')[1])
