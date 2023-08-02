# Importing the required library
import pandas_datareader as pdr

# Setting the beginning and end of the historical data
start_date = '1990-01-01'
end_date   = '2023-01-23'

# Creating a dataframe and downloading the VIX data
vix = pdr.DataReader('VIXCLS', 'fred', start_date, end_date)

# Printing the latest five observations of the dataframe
print(vix.tail())

# Calculating the number of nan values
count_nan = vix['VIXCLS'].isnull().sum()

# Printing the result
print('Number of nan values in the VIX dataframe: ' + str(count_nan))

# Dropping the NaN values from the rows
vix = vix.dropna()

# Taking the differences in an attempt to make the data stationary
vix = vix.diff(periods = 1, axis = 0)

# Dropping the first value of the data frame
vix = vix.iloc[1: , :]

# Calculating the mean of the dataset
mean = vix["VIXCLS"].mean()

# Printing the result
print('The mean of the dataset = ' + str(mean))

# Importing the required library
import matplotlib.pyplot as plt

# Plotting the latest 250 observations in black with a label
plt.plot(vix[-250:], color = 'black', linewidth = 1.5, 
         label = 'Change in VIX')

# Plotting a red dashed horizontal line that is equal to mean
plt.axhline(y = mean, color = 'red', linestyle = 'dashed')

# Calling a grid to facilitate the visual component
plt.grid()

# Calling the legend function so it appears with the chart
plt.legend()

# Calling the plot
plt.show()
