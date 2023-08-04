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

# Calculating the variance of the dataset
variance = cpi_latest["CPIAUCSL"].var() 

# Printing the result
print('The variance of the dataset: ' + str(variance), '%')

# Calculating the standard deviation of the dataset
standard_deviation = cpi_latest["CPIAUCSL"].std() 

# Printing the result
print('The standard deviation of the dataset: ' + str(standard_deviation), '%')

# Calculating the range of the dataset
range_metric = max(cpi["CPIAUCSL"]) - min(cpi["CPIAUCSL"])

# Printing the result
print('The range of the dataset: ' + str(range_metric), '%')

