
from fracdiff.sklearn import Fracdiff
import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt

# Set the start and end dates for the data
start_date = '1990-01-01'
end_date   = '2023-06-01'

# Fetch S&P 500 price data
spx = pdr.get_data_fred('SP500', start = start_date, end = end_date)
spx = spx.dropna()
spx = np.array(spx)

# Calculate the fractional differentiation
window = 100
f = Fracdiff(0.50, mode = "valid", window = window)
frac_spx = f.fit_transform(spx)

# Calculate a simple differencing function for comparison
diff_spx = np.diff(spx[:, 0])
diff_spx = np.reshape(diff_spx, (-1, 1))

# Harmonizing time indices
spx = spx[window - 1:, ]
diff_spx = diff_spx[window - 2:, ]

# Plotting
fig, axes = plt.subplots(nrows = 3, ncols = 1)

axes[0].plot(spx[5:,], label = 'S&P 500', color = 'blue', linewidth = 1)
axes[1].plot(frac_spx[5:,], label = 'Fractionally Differentiated S&P 500 (0.50)', color = 'orange', linewidth = 1)
axes[2].plot(diff_spx[5:,], label = 'Differenced S&P 500', color = 'green', linewidth = 1)

axes[0].legend()
axes[1].legend()
axes[2].legend()

axes[0].grid()
axes[1].grid()
axes[2].grid()   

axes[1].axhline(y = 0, color = 'black', linestyle = 'dashed') 
axes[2].axhline(y = 0, color = 'black', linestyle = 'dashed')  

# Checking for stationarity
from statsmodels.tsa.stattools import adfuller

adfuller(spx) 
print('p-value: %f' % adfuller(spx)[1])

adfuller(frac_spx) 
print('p-value: %f' % adfuller(frac_spx)[1])

adfuller(diff_spx) 
print('p-value: %f' % adfuller(diff_spx)[1])
