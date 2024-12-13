import numpy as np
import pandas as pd

def compute_technical_indicators(df):
    """
    Compute technical indicators (SMA, RSI, Bollinger Bands) for stock data.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data (Date, Open, High, Low, Close, Volume).
        
    Returns:
        pd.DataFrame: Updated dataframe with additional columns for technical indicators.
    """
    # Validate input
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")
    
    # Check for required column
    if 'Close' not in df.columns:
        raise ValueError("'Close' column is missing from the DataFrame.")
    
    # Ensure 'Close' column is a Series
    if not isinstance(df['Close'], (pd.Series, list, tuple, np.ndarray)):
        raise TypeError("'Close' column must be a Series, list, tuple, or 1-d array.")
    
    # Ensure 'Close' column is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    if df['Close'].isnull().all():
        raise ValueError("'Close' column contains no valid data.")
    
    # Simple Moving Average (SMA)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Bollinger Bands
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_Mid'] = df['SMA_20']
    df['BB_Upper'] = df['SMA_20'] + (rolling_std * 2)
    df['BB_Lower'] = df['SMA_20'] - (rolling_std * 2)
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df
