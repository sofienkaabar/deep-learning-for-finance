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

# Calculating correlation
combined_cpi_data.corr(method = 'pearson')
combined_cpi_data.corr(method = 'spearman')

# Creating a dataframe and downloading the CPI data using its code name and its source
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

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
from minepy import MINE # Must be pip installed (pip install minepy --user)

# Calculating the linear correlation measures
print('Correlation | Pearson: ', round(pearsonr(sine, cosine)[0], 3))
print('Correlation | Spearman: ', round(spearmanr(sine, cosine)[0], 3))

# Calculating the MIC
mine = MINE(alpha = 0.6, c = 15)
mine.compute_score(sine,cosine)
MIC = mine.mic()
print('Correlation | MIC: ', round(MIC, 3))
