# Importing the required library
import pandas_datareader as pdr

# Setting the beginning and end of the historical data
start_date = '1950-01-01'
end_date   = '2023-01-23'

# Creating a dataframe and downloading the CPI data using its code name and its source
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Printing the latest five observations of the dataframe
print(cpi.tail())

# Importing the required library
import pandas as pd

# Checking if there are NaN values in the CPI dataframe previously defined
count_nan = cpi['CPIAUCSL'].isnull().sum()

# Printing the result
print('Number of NaN values in the CPI dataframe: ' + str(count_nan))

# Dropping the NaN values from the rows
cpi = cpi.dropna()

# Transforming the CPI into a year-on-year measure
cpi = cpi.pct_change(periods = 12, axis = 0) * 100
cpi = cpi.dropna()

# Calculating the mean of the CPI over the last 10 years
cpi_last_ten_years = cpi.iloc[-240:]
mean = cpi_last_ten_years["CPIAUCSL"].mean()

# Printing the result
print('The mean of the dataset: ' + str(mean), '%')

# Importing the required library
import matplotlib.pyplot as plt

# Plotting the latest observations in black with a label
plt.plot(cpi_last_ten_years[:], color = 'black', linewidth = 1.5, label = 'Change in CPI Year-on-Year')

# Plotting horizontal lines that represent the mean and the zero threshold
plt.axhline(y = mean, color = 'red', linestyle = 'dashed', label = '10-Year Mean')
plt.axhline(y = 0, color = 'blue', linestyle = 'dashed', linewidth = 1)

# Calling a grid to facilitate the visual component
plt.grid()

# Calling the legend function so it appears with the chart
plt.legend()

# Calculating the median of the dataset
median = cpi_last_ten_years["CPIAUCSL"].median() 

# Printing the result
print('The median of the dataset: ' + str(median), '%')

# Plotting the latest observations in black with a label
plt.plot(cpi_last_ten_years[:], color = 'black', linewidth = 1.5, label = 'Change in CPI Year-on-Year')

plt.axhline(y = median, color = 'red', linestyle = 'dashed', label = '10-Year Median')
plt.axhline(y = 0, color = 'blue', linestyle = 'dashed', linewidth = 1)

# Calling a grid to facilitate the visual component
plt.grid()

# Calling the legend function so it appears with the chart
plt.legend()

# Calculating the variance of the dataset
variance = cpi_last_ten_years["CPIAUCSL"].var() 

# Printing the result
print('The variance of the dataset: ' + str(variance), '%')

# Calculating the standard deviation of the dataset
standard_deviation = cpi_last_ten_years["CPIAUCSL"].std() 

# Printing the result
print('The standard deviation of the dataset: ' + str(standard_deviation), '%')

# Calculating the range of the dataset
range_metric = max(cpi["CPIAUCSL"]) - min(cpi["CPIAUCSL"])

# Printing the result
print('The range of the dataset: ' + str(range_metric), '%')

# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Generate data for the plot
data = np.linspace(-3, 3, num = 1000)

# Define the mean and standard deviation of the normal distribution
mean = 0
std = 1

# Generate the PDF of the normal distribution
pdf = stats.norm.pdf(data, mean, std)

# Plot the normal distribution plot
plt.plot(data, pdf, '-', color = 'black', lw = 2)
plt.axvline(mean, color = 'black', linestyle = '--')

# Calling a grid to facilitate the visual component
plt.grid()

# Show the plot
plt.show()

# Calculating the skew of the dataset
skew = cpi["CPIAUCSL"].skew() 

# Printing the result
print('The skew of the dataset: ' + str(skew))

# Plotting the histogram of the data
fig, ax = plt.subplots()
ax.hist(cpi['CPIAUCSL'], bins = 30, edgecolor = 'black', color = 'white')

# Add vertical lines for better interpretation
ax.axvline(mean, color='black', linestyle='--', label='Mean', linewidth = 2)
ax.axvline(median, color='grey', linestyle='-.', label='Median', linewidth = 2)

# Calling the grid function for better interpretability
plt.grid()

# Calling the legend function to show the labels
plt.legend()

# Showing the plot
plt.show()

# Calculating the excess kurtosis of the dataset
excess_kurtosis = cpi["CPIAUCSL"].kurtosis() 

# Printing the result
print('The excess kurtosis of the dataset: ' + str(excess_kurtosis))

# Importing the required library
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt

# Setting the beginning and end of the historical data
start_date = '1950-01-01'
end_date   = '2022-12-01'

# Creating a dataframe and downloading the CPI data using its code name and its source
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Transforming the CPI into a year-on-year measure
cpi = cpi.pct_change(periods = 12, axis = 0) * 100

# Dropping the NaN values
cpi = cpi.dropna()

# Resetting the index
cpi = cpi.reset_index()

# Creating the chart
fig, ax = plt.subplots()
ax.scatter(cpi['DATE'], cpi['CPIAUCSL'], color = 'black', s = 8,  label = 'Change in CPI Year-on-Year')

# Calling the grid function for better interpretability
plt.grid()

# Calling the legend function to show the labels
plt.legend()

# Showing the plot
plt.show()

# Setting the beginning and end of the historical data
start_date = '1995-01-01'
end_date   = '2022-12-01'

# Creating a dataframe and downloading the CPI data using its code name and its source
cpi_us = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)
cpi_uk = pdr.DataReader('GBRCPIALLMINMEI', 'fred', start_date, end_date)

# Dropping the NaN values from the rows
cpi_us = cpi_us.dropna()
cpi_uk = cpi_uk.dropna()

# Transforming the CPI into a year-on-year measure
cpi_us = cpi_us.pct_change(periods = 12, axis = 0) * 100
cpi_us = cpi_us.dropna()

cpi_uk = cpi_uk.pct_change(periods = 12, axis = 0) * 100
cpi_uk = cpi_uk.dropna()

# Creating the chart
fig, ax = plt.subplots()
ax.scatter(cpi_us['CPIAUCSL'], cpi_uk['GBRCPIALLMINMEI'], color = 'black', s = 8, label = 'Change in CPI Year-on-Year')

# Adding a few aesthetic elements to the chart
ax.set_xlabel('US CPI')
ax.set_ylabel('UK CPI')
ax.axvline(x = 0, color='black', linestyle = 'dashed', linewidth = 1)  # vertical line
ax.axhline(y = 0, color='black', linestyle = 'dashed', linewidth = 1)  # horizontal line
ax.set_ylim(-2,)

# Calling the grid function for better interpretability
plt.grid()

# Calling the legend function to show the labels
plt.legend()

# Showing the plot
plt.show()

# Setting the beginning and end of the historical data
start_date = '1950-01-01'
end_date   = '2022-12-01'

# Creating a dataframe and downloading the CPI data using its code name and its source
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Transforming the CPI into a year-on-year measure
cpi = cpi.pct_change(periods = 12, axis = 0) * 100

# Dropping the NaN values
cpi = cpi.dropna()

# Resetting the index
cpi = cpi.reset_index()

# Creating the chart
plt.plot(cpi['DATE'], cpi['CPIAUCSL'], color = 'black', label = 'Change in CPI Year-on-Year')

# Calling the grid function for better interpretability
plt.grid()

# Calling the legend function to show the labels
plt.legend()

# Showing the plot
plt.show()

# Taking the values of the previous twelve months
cpi_one_year = cpi.iloc[-12:]

# Creating the chart
plt.bar(cpi_one_year['DATE'], cpi_one_year['CPIAUCSL'], color = 'black', label = 'Change in CPI Year-on-Year', width = 7)

# Calling the grid function for better interpretability
plt.grid()

# Calling the legend function to show the labels
plt.legend()

# Showing the plot
plt.show()

# Creating the chart
fig, ax = plt.subplots()
ax.hist(cpi['CPIAUCSL'], bins = 30, edgecolor = 'black', color = 'white', label = 'Change in CPI Year-on-Year',)

# Add vertical lines for better interpretation
ax.axvline(0, color='black')

# Calling the grid function for better interpretability
plt.grid()

# Calling the legend function to show the labels
plt.legend()

# Showing the plot
plt.show()

# Taking the values of the last twenty years
cpi_last_ten_years = cpi.iloc[-240:]

# Creating the chart
fig, ax = plt.subplots()
ax.boxplot(cpi_last_ten_years['CPIAUCSL'])

# Calling the grid function for better interpretability
plt.grid()

# Calling the legend function to show the labels
plt.legend()

# Showing the plot
plt.show()

# Replace the corresponding code line with the following
ax.boxplot(cpi_last_ten_years['CPIAUCSL'], showfliers = False)

# Importing the required libraries
import pandas_datareader as pdr
import pandas as pd

# Setting the beginning and end of the historical data
start_date = '1995-01-01'
end_date   = '2022-12-01'

# Creating a dataframe and downloading the CPI data using its code name and its source
cpi_us = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)
cpi_uk = pdr.DataReader('GBRCPIALLMINMEI', 'fred', start_date, end_date)

# Dropping the NaN values from the rows
cpi_us = cpi_us.dropna()
cpi_uk = cpi_uk.dropna()

# Transforming the US CPI into a year-on-year measure
cpi_us = cpi_us.pct_change(periods = 12, axis = 0) * 100
cpi_us = cpi_us.dropna()

# Transforming the UK CPI into a year-on-year measure
cpi_uk = cpi_uk.pct_change(periods = 12, axis = 0) * 100
cpi_uk = cpi_uk.dropna()

# Joining both CPI data into one dataframe
combined_cpi_data = pd.concat([cpi_us['CPIAUCSL'], cpi_uk['GBRCPIALLMINMEI']], axis = 1)

# Using pandas' correlation function to calculate the measure
combined_cpi_data.corr(method = 'pearson')

# Importing the required libraries
import pandas_datareader as pdr
import pandas as pd

# Setting the beginning and end of the historical data
start_date = '1995-01-01'
end_date   = '2022-12-01'

# Creating a dataframe and downloading the CPI data using its code name and its source
cpi_us = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)
cpi_uk = pdr.DataReader('GBRCPIALLMINMEI', 'fred', start_date, end_date)

# Dropping the NaN values from the rows
cpi_us = cpi_us.dropna()
cpi_uk = cpi_uk.dropna()

# Transforming the US CPI into a year-on-year measure
cpi_us = cpi_us.pct_change(periods = 12, axis = 0) * 100
cpi_us = cpi_us.dropna()

# Transforming the UK CPI into a year-on-year measure
cpi_uk = cpi_uk.pct_change(periods = 12, axis = 0) * 100
cpi_uk = cpi_uk.dropna()

# Joining both CPI data into one dataframe
combined_cpi_data = pd.concat([cpi_us['CPIAUCSL'], cpi_uk['GBRCPIALLMINMEI']], axis = 1)

# Using pandas' correlation function to calculate the measure
combined_cpi_data.corr(method = 'spearman')

# Importing the required libraries
import pandas_datareader as pdr
import pandas as pd

# Setting the beginning and end of the historical data
start_date = '1950-01-01'
end_date   = '2022-12-01'

# Creating a dataframe and downloading the CPI data using its code name and its source
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Dropping the NaN values from the rows
cpi = cpi.dropna()

# Transforming the US CPI into a year-on-year measure
cpi = cpi.pct_change(periods = 12, axis = 0) * 100
cpi = cpi.dropna()

# Transforming the data frame to a series structure
cpi = cpi.iloc[:,0]

# Calculating autocorrelation with a lag of 1
print('Correlation with a lag of 1 = ', round(cpi.autocorr(lag = 1), 2))

# Calculating autocorrelation with a lag of 6
print('Correlation with a lag of 6 = ', round(cpi.autocorr(lag = 6), 2))

# Calculating autocorrelation with a lag of 12
print('Correlation with a lag of 12 = ', round(cpi.autocorr(lag = 12), 2))

# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt

# Setting the range of the data
data_range = np.arange(0, 30, 0.1)

# Creating the sine and the cosine waves
sine = np.sin(data_range)
cosine = np.cos(data_range)

# Plotting
plt.plot(sine, color = 'black', label = 'Sine Function')
plt.plot(cosine, color = 'grey', linestyle = 'dashed', label = 'Cosine Function')
plt.grid()
plt.legend()

# Importing the libraries
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from minepy import MINE

# Calculating the linear correlation measures
print('Correlation | Pearson: ', round(pearsonr(sine, cosine)[0], 3))
print('Correlation | Spearman: ', round(spearmanr(sine, cosine)[0], 3))

# Calculating the MIC
mine = MINE(alpha = 0.6, c = 15)
mine.compute_score(sine,cosine)
MIC = mine.mic()
print('Correlation | MIC: ', round(MIC, 3))

# Importing the required libraries
from statsmodels.tsa.stattools import adfuller
import pandas_datareader as pdr

# Setting the beginning and end of the historical data
start_date = '1950-01-01'
end_date   = '2022-12-01'

# Creating a dataframe and downloading the CPI data using its code name and its source
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Dropping the NaN values from the rows
cpi = cpi.dropna()

# Transforming the US CPI into a year-on-year measure
cpi = cpi.pct_change(periods = 12, axis = 0) * 100
cpi = cpi.dropna()

# Applying the ADF test on the CPI data
adfuller(cpi) 
print('p-value: %f' % adfuller(cpi)[1])

# Importing the required libraries
from statsmodels.tsa.stattools import adfuller
import pandas_datareader as pdr

# Setting the beginning and end of the historical data
start_date = '1950-01-01'
end_date   = '2022-12-01'

# Creating a dataframe and downloading the CPI data using its code name and its source
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Dropping the NaN values from the rows
cpi = cpi.dropna()

# Applying the ADF test on the CPI data
adfuller(cpi) 
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
kpss(sinewave) 
print('p-value: %f' % kpss(sinewave)[1])

# KPSS testing | Ascending sine wave
kpss(sinewave_ascending) 
print('p-value: %f' % kpss(sinewave_ascending)[1])

# KPSS testing while taking into account the trend | Ascending sine wave
kpss(sinewave_ascending, regression = 'ct') 
print('p-value: %f' % kpss(sinewave_ascending, regression = 'ct')[1])

# Importing the required libraries
from statsmodels.tsa.stattools import kpss
import pandas_datareader as pdr

# Setting the beginning and end of the historical data
start_date = '1950-01-01'
end_date   = '2022-12-01'

# Creating a dataframe and downloading the CPI data using its code name and its source
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Dropping the NaN values from the rows
cpi = cpi.dropna()

# Transforming the US CPI into a year-on-year measure
cpi = cpi.pct_change(periods = 12, axis = 0) * 100
cpi = cpi.dropna()

# Applying the KPSS (no trend consideration) test on the CPI data
kpss(cpi) 
print('p-value: %f' % kpss(cpi)[1])

# Applying the KPSS (with trend consideration) test on the CPI data
kpss(cpi, regression = 'ct') 
print('p-value: %f' % kpss(cpi, regression = 'ct')[1])















