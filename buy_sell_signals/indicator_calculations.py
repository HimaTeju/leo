from ta.momentum import RSIIndicator
from ta.trend import MACD
import pandas as pd
from data_loader import load_data

def calculate_rsi(data, period=14):
    close_prices = data['Close'].squeeze()
    
    # Ensure there are enough data points for RSI calculation
    if len(close_prices) < period:
        print(f"Not enough data to calculate RSI. Need at least {period} data points.")
        return pd.Series([None] * len(close_prices), index=close_prices.index)
    
    # Calculate RSI
    rsi = RSIIndicator(close_prices, window=period).rsi()
    

    return rsi

def calculate_macd(data):
    close_prices = data['Close'].squeeze() 
    
    # Ensure there are enough data points for MACD calculation
    if len(close_prices) < 26:  # MACD typically requires at least 26 data points
        print("Not enough data to calculate MACD. Need at least 26 data points.")
        return pd.Series([None] * len(close_prices), index=close_prices.index), pd.Series([None] * len(close_prices), index=close_prices.index)
    
    # Calculate MACD and Signal Line
    macd = MACD(close_prices)
    macd_line = macd.macd()
    signal_line = macd.macd_signal()
    
    return macd_line, signal_line

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    if len(data) < window:
        print(f"Not enough data to calculate Bollinger Bands. Need at least {window} data points.")
        return pd.Series([None] * len(data), index=data.index), pd.Series([None] * len(data), index=data.index), pd.Series([None] * len(data), index=data.index)
    
    # Calculate the rolling mean (Middle Band)
    middle_band = data['Close'].rolling(window=window, min_periods=1).mean()
    
    # Calculate the rolling standard deviation
    rolling_std_dev = data['Close'].rolling(window=window, min_periods=1).std()
    
    # Calculate the Upper and Lower Bands
    upper_band = middle_band + (rolling_std_dev * num_std_dev)
    lower_band = middle_band - (rolling_std_dev * num_std_dev)
    
    return upper_band, middle_band, lower_band
