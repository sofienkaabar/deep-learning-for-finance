
from fracdiff.sklearn import Fracdiff
import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt

# Set the start and end dates for the data
start_date = '2005-01-01'
end_date   = '2023-06-01'

# Fetch EURUSD price data
eurusd = pdr.get_data_fred('DEXUSEU', start = start_date, end = end_date)
eurusd = eurusd.dropna()
eurusd = np.array(eurusd)

# Calculate the fractional differentiation
window = 100
f = Fracdiff(0.30, mode = "valid", window = window)
frac_eurusd = f.fit_transform(eurusd)

# Calculate a simple differencing function for comparison
diff_eurusd = np.diff(eurusd[:, 0])
diff_eurusd = np.reshape(diff_eurusd, (-1, 1))

# Harmonizing time indices
eurusd = eurusd[window - 1:, ]
diff_eurusd = diff_eurusd[window - 2:, ]

# Plotting
fig, axes = plt.subplots(nrows = 3, ncols = 1)

axes[0].plot(eurusd[5:,], label = 'EURUSD', color = 'blue', linewidth = 1)
axes[1].plot(frac_eurusd[5:,], label = 'Fractionally Differentiated EURUSD (0.30)', color = 'orange', linewidth = 1)
axes[2].plot(diff_eurusd[5:,], label = 'Differenced EURUSD', color = 'green', linewidth = 1)

axes[0].legend()
axes[1].legend()
axes[2].legend()

axes[0].grid()
axes[1].grid()
axes[2].grid()   

axes[2].axhline(y = 0, color = 'black', linestyle = 'dashed')  

# Checking for stationarity
from statsmodels.tsa.stattools import adfuller

adfuller(eurusd) 
print('p-value: %f' % adfuller(eurusd)[1])

adfuller(frac_eurusd) 
print('p-value: %f' % adfuller(frac_eurusd)[1])

adfuller(diff_eurusd) 
print('p-value: %f' % adfuller(diff_eurusd)[1])
