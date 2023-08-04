# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Generate data for the plot
data = np.linspace(-3, 3, num = 1000)

# Define the mean and standard deviation of the normal distribution
mean = 0
std = 1

# Generate the function of the normal distribution
pdf = stats.norm.pdf(data, mean, std)

# Plot the normal distribution plot
plt.plot(data, pdf, '-', color = 'black', lw = 2)
plt.axvline(mean, color = 'black', linestyle = '--')

# Calling a grid to facilitate the visual component
plt.grid()

# Show the plot
plt.show()

# Importing the required library
import pandas_datareader as pdr

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

# Isolating the last 20 years data
cpi_latest = cpi.iloc[-240:]

# Calculating the skew of the dataset
skew = cpi_latest["CPIAUCSL"].skew() 

# Recalling the mean and median for the histogram
mean = cpi_latest["CPIAUCSL"].mean()
median = cpi_latest["CPIAUCSL"].median() 

# Printing the result
print('The skew of the dataset: ' + str(skew))

# Plotting the histogram of the data
fig, ax = plt.subplots()
ax.hist(cpi_latest['CPIAUCSL'], bins = 30, edgecolor = 'black', color = 'white')

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
excess_kurtosis = cpi_latest["CPIAUCSL"].kurtosis() 

# Printing the result
print('The excess kurtosis of the dataset: ' + str(excess_kurtosis))