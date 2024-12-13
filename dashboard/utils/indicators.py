import pandas as pd
import numpy as np

def compute_technical_indicators(df):
    """
    Compute technical indicators (SMA, RSI, Bollinger Bands) for stock data.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data (Date, Open, High, Low, Close, Volume).
        
    Returns:
        pd.DataFrame: Updated dataframe with additional columns for technical indicators.
    """
    # Simple Moving Average (SMA)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Mid'] = df['SMA_20']
    df['BB_Upper'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
    
    return df
