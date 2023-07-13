import datetime
import pytz
import pandas                    as pd
import MetaTrader5               as mt5
import matplotlib.pyplot         as plt
import numpy                     as np

frame_M15  = mt5.TIMEFRAME_M15
frame_M30  = mt5.TIMEFRAME_M30
frame_H1   = mt5.TIMEFRAME_H1
frame_D1   = mt5.TIMEFRAME_D1
frame_W1   = mt5.TIMEFRAME_W1
frame_M1   = mt5.TIMEFRAME_MN1

now = datetime.datetime.now()

assets = ['EURUSD', 'USDCHF', 'GBPUSD', 'USDCAD', 'AUDUSD', 'NZDUSD', 'EURGBP', 'EURCHF', 'EURCAD', 'EURAUD']
     
def mass_import(asset, time_frame):

    if time_frame == 'M15':
        data = get_quotes(frame_M15, 2023, 6, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)    

    if time_frame == 'M30':
        data = get_quotes(frame_M30, 2023, 6, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)    
                
    if time_frame == 'H1':
        data = get_quotes(frame_H1, 2014, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        
        
    if time_frame == 'D1':
        data = get_quotes(frame_D1, 2003, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        
 
    if time_frame == 'W1':
        data = get_quotes(frame_W1, 2002, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        
 
    if time_frame == 'M1':
        data = get_quotes(frame_M1, 2000, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)
                  
    return data 

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