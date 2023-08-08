
'''
↓↓↓↓↓↓↓↓↓IMPORTANT READ FIRST↓↓↓↓↓↓↓↓↓
||||||||||||||||||||||||||||||||||||||||

To properly use this file, please use the following guidelines:
    
    1. Put this file in the directory used by the interpreter
    2. In SPYDER, the directory is generally on the top right
    3. Alternatively, you can open this file and execute it

PUT THIS FILE IN THE PYTHON DIRECTORY IN ORDER TO PROPERLY IMPORT ITS FUNCTIONS

||||||||||||||||||||||||||||||||||||||||
↑↑↑↑↑↑↑↑↑IMPORTANT READ FIRST↑↑↑↑↑↑↑↑↑
'''

import datetime
import pytz
import pandas                    as pd
import MetaTrader5               as mt5
import matplotlib.pyplot         as plt
import numpy                     as np
import cot_reports               as cot
import requests
import json  

now = datetime.datetime.now()

assets = ['EURUSD', 'USDCHF', 'GBPUSD', 'USDCAD', 'AUDUSD', 'NZDUSD', 'EURGBP', 'EURCHF', 'EURCAD', 'EURAUD']
 
def get_quotes(time_frame, year = 2005, month = 1, day = 1, asset = "EURUSD"):    
    if not mt5.initialize(): 
        print("initialize() failed, error code =", mt5.last_error()) 
        quit()
    timezone = pytz.timezone("Europe/Paris")
    time_from = datetime.datetime(year, month, day, tzinfo = timezone)
    time_to = datetime.datetime.now(timezone) + datetime.timedelta(days=1)
    rates = mt5.copy_rates_range(asset, time_frame, time_from, time_to)
    rates_frame = pd.DataFrame(rates)
    
    return rates_frame    

def mass_import(asset, time_frame):
    if time_frame == 'M15':
        data = get_quotes(mt5.TIMEFRAME_M15, 2023, 6, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)    
    if time_frame == 'M30':
        data = get_quotes(mt5.TIMEFRAME_M30, 2023, 6, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)              
    if time_frame == 'H1':
        data = get_quotes(mt5.TIMEFRAME_H1, 2015, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)         
    if time_frame == 'D1':
        data = get_quotes(mt5.TIMEFRAME_D1, 2003, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        
    if time_frame == 'W1':
        data = get_quotes(mt5.TIMEFRAME_W1, 2002, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        
    if time_frame == 'M1':
        data = get_quotes(mt5.TIMEFRAME_MN1, 2000, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)             
    
    return data 

def data_preprocessing(data, num_lags, train_test_split):
    # Prepare the data for training
    x = []
    y = []
    for i in range(len(data) - num_lags):
        x.append(data[i:i + num_lags])
        y.append(data[i+ num_lags])
    # Convert the data to numpy arrays
    x = np.array(x)
    y = np.array(y)
    # Split the data into training and testing sets
    split_index = int(train_test_split * len(x))
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]

    return x_train, y_train, x_test, y_test

def recursive_mpf(x_test, y_test, num_lags, model):
    # Latest values to use as inputs
    x_test = x_test[-1]
    x_test = np.reshape(x_test, (-1, 1))
    x_test = np.transpose(x_test)
    x_test = x_test.reshape((-1, num_lags, 1))
    y_predicted = []
    for i in range(len(y_test)):     
        # Predict over the last x_test values
        predicted_value = model.predict(x_test)
        y_predicted = np.append(y_predicted, predicted_value)
        # Re-inserting the latest prediction into x_test array
        x_test = np.transpose(x_test)
        x_test = np.append(x_test, predicted_value)
        x_test = x_test[1:, ]
        x_test = np.reshape(x_test, (-1, 1))
        x_test = np.transpose(x_test)
        x_test = x_test.reshape((-1, num_lags, 1))  
    y_predicted = np.reshape(y_predicted, (-1, 1))
        
    return x_test, y_predicted

def direct_mpf(data, time_steps, train_test_split, forecast_horizon):
    x, y = [], []
    for i in range(len(data) - time_steps - forecast_horizon + 1):
        x.append(data[i:i + time_steps])
        y.append(data[i + time_steps:i + time_steps + forecast_horizon])
    x = np.array(x)
    y = np.array(y)   
    split_index = int(train_test_split * len(x))
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:] 
    
    return x_train, y_train, x_test, y_test

def import_cot_data(start_year, end_year, market):
    df = pd.DataFrame()
    for i in range(start_year, end_year + 1):
        single_year = pd.DataFrame(cot.cot_year(i, cot_report_type='traders_in_financial_futures_fut')) 
        df = df.append(single_year, ignore_index=True)
    new_df = df.loc[:, ['Market_and_Exchange_Names',
                        'Report_Date_as_YYYY-MM-DD',
                        'Pct_of_OI_Dealer_Long_All',
                        'Pct_of_OI_Dealer_Short_All',
                        'Pct_of_OI_Lev_Money_Long_All',                    
                        'Pct_of_OI_Lev_Money_Short_All']]
    new_df['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(new_df['Report_Date_as_YYYY-MM-DD'])
    new_df = new_df.sort_values(by='Report_Date_as_YYYY-MM-DD')
    data = new_df[new_df['Market_and_Exchange_Names'] == market]
    data['Net_COT'] = (data['Pct_of_OI_Lev_Money_Long_All'] - \
                       data['Pct_of_OI_Lev_Money_Short_All']) - \
                      (data['Pct_of_OI_Dealer_Long_All'] -\
                       data['Pct_of_OI_Dealer_Short_All'])                
    
    return data

def plot_train_test_values(window, train_window, y_train, y_test, y_predicted):
    prediction_window = window
    first = train_window
    second = window - first
    y_predicted = np.reshape(y_predicted, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))
    plotting_time_series = np.zeros((prediction_window, 3))
    plotting_time_series[0:first, 0] = y_train[-first:]
    plotting_time_series[first:, 1] = y_test[0:second, 0]
    plotting_time_series[first:, 2] = y_predicted[0:second, 0] 
    plotting_time_series[0:first, 1] = plotting_time_series[0:first, 1] / 0
    plotting_time_series[0:first, 2] = plotting_time_series[0:first, 2] / 0
    plotting_time_series[first:, 0] = plotting_time_series[first:, 0] / 0
    plt.plot(plotting_time_series[:, 0], label = 'Training data', color = 'black', linewidth = 2.5)
    plt.plot(plotting_time_series[:, 1], label = 'Test data', color = 'black', linestyle = 'dashed', linewidth = 2)
    plt.plot(plotting_time_series[:, 2], label = 'Predicted data', color = 'red', linewidth = 1)
    plt.axvline(x = first, color = 'black', linestyle = '--', linewidth = 1)
    plt.grid()
    plt.legend()

def calculate_accuracy(predicted_returns, real_returns):
    predicted_returns = np.reshape(predicted_returns, (-1, 1))
    real_returns = np.reshape(real_returns, (-1, 1))
    hits = sum((np.sign(predicted_returns)) == np.sign(real_returns))
    total_samples = len(predicted_returns)
    accuracy = hits / total_samples
    
    return accuracy[0] * 100

def model_bias(predicted_returns):
    bullish_forecasts = np.sum(predicted_returns > 0)
    bearish_forecasts = np.sum(predicted_returns < 0)
    
    return bullish_forecasts / bearish_forecasts

def calculate_directional_accuracy(predicted_returns, real_returns):
    # Calculate differences between consecutive elements
    diff_predicted = np.diff(predicted_returns, axis = 0)
    diff_real = np.diff(real_returns, axis = 0)
    # Check if signs of differences are the same
    store = []  
    for i in range(len(predicted_returns)):
        try:            
            if np.sign(diff_predicted[i]) == np.sign(diff_real[i]):                
                store = np.append(store, 1)        
            elif np.sign(diff_predicted[i]) != np.sign(diff_real[i]):                
                store = np.append(store, 0)                  
        except IndexError:           
            pass       
    directional_accuracy = np.sum(store) / len(store)
        
    return directional_accuracy * 100

def import_crypto(symbol, interval = '1h'): 
    # Getting the original link from Binance
    url = 'https://api.binance.com/api/v1/klines'
    # Linking the link with the Cryptocurrency and the time frame
    link = url + '?symbol=' + symbol + '&interval=' + interval
    # Requesting the data in the form of text
    data = json.loads(requests.get(link).text)
    # Converting the text data to dataframe
    data = np.array(data)
    data = data.astype(np.float)
    data = data[:, 1:5]
    
    return data

def multiple_data_preprocessing(data, train_test_split):
    data = add_column(data, 4)
    data[:, 1] = np.roll(data[:, 1], 1, axis = 0)
    data[:, 2] = np.roll(data[:, 2], 1, axis = 0)
    data[:, 3] = np.roll(data[:, 1], 1, axis = 0)
    data[:, 4] = np.roll(data[:, 2], 1, axis = 0)
    data[:, 5] = np.roll(data[:, 3], 1, axis = 0)
    data[:, 6] = np.roll(data[:, 4], 1, axis = 0)
    data = data[1:, ]
    x = data[:, 1:]
    y = data[:, 0]
    split_index = int(train_test_split * len(x))
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]
    
    return x_train, y_train, x_test, y_test

def volatility(data, lookback, close, position):
    data = add_column(np.reshape(data, (-1, 1)), 1)
    for i in range(len(data)):   
        try:           
            data[i, position] = (data[i - lookback + 1:i + 1, close].std())  
        except IndexError:          
            pass   
    data = delete_row(data, lookback)    
     
    return data

def add_column(data, times): 
    for i in range(1, times + 1): 
        new = np.zeros((len(data), 1), dtype = float)     
        data = np.append(data, new, axis = 1)
        
    return data

def delete_column(data, index, times):  
    for i in range(1, times + 1):   
        data = np.delete(data, index, axis = 1)

    return data

def delete_row(data, number): 
    data = data[number:, ]
    
    return data

def compute_diff(data, period):
    data = add_column(np.reshape(data, (-1, 1)), 1)
    for i in range(len(data)):
        data[i, -1] = data[i, 0] - data[i - 1, 0]
    data = delete_column(data, 0, 1)
    
    return data

def ma(data, lookback, close, position):     
    data = add_column(data, 1)    
    for i in range(len(data)):           
            try:                
                data[i, position] = (data[i - lookback + 1:i + 1, close].mean())            
            except IndexError:               
                pass           
    data = delete_row(data, lookback)
    
    return data

def smoothed_ma(data, alpha, lookback, close, position):    
    lookback = (2 * lookback) - 1    
    alpha = alpha / (lookback + 1.0)    
    beta  = 1 - alpha    
    data = ma(data, lookback, close, position)
    data[lookback + 1, position] = (data[lookback + 1, close] * alpha) + (data[lookback, position] * beta)
    for i in range(lookback + 2, len(data)):
            try:
                data[i, position] = (data[i, close] * alpha) + (data[i - 1, position] * beta)
            except IndexError:
                pass
            
    return data

def rsi(data, lookback, close, position):
    data = add_column(data, 5)
    for i in range(len(data)): 
        data[i, position] = data[i, close] - data[i - 1, close]
    for i in range(len(data)):
        if data[i, position] > 0:
            data[i, position + 1] = data[i, position]
        elif data[i, position] < 0:       
            data[i, position + 2] = abs(data[i, position])         
    data = smoothed_ma(data, 2, lookback, position + 1, position + 3)
    data = smoothed_ma(data, 2, lookback, position + 2, position + 4)
    data[:, position + 5] = data[:, position + 3] / data[:, position + 4]   
    data[:, position + 6] = (100 - (100 / (1 + data[:, position + 5])))
    data = delete_column(data, position, 6)
    data = delete_row(data, lookback)

    return data