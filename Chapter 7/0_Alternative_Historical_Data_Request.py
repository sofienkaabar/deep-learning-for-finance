'''
FIRST METHOD
'''

# Importing the required libraries
import pandas_datareader as pdr
import numpy as np

# Set the start and end dates for the data
start_date = '2000-01-01'
end_date   = '2023-06-01'

# Fetch EURUSD price data
data = np.array((pdr.get_data_fred('DEXUSEU', start = start_date, end = end_date)).dropna())

# Difference the data and make it stationary
data = np.diff(data[:, 0])

# For EURUSD, use 'DEXUSEU'
# For GBPUSD, use 'DEXUSUK'
# For USDJPY, use 'DEXJPUS'
# For S&P 500, use 'SP500'

'''
SECOND METHOD
'''

# Importing the required libraries
import pandas as pd
import numpy as np

# Import the data
data = np.array(pd.read_excel('Daily_EURUSD_Historical_Data.xlsx')['<CLOSE>'])

# Difference the data and make it stationary
data = np.diff(data)
