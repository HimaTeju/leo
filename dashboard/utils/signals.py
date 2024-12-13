import numpy as np

def generate_signals(df):
    """
    Generate buy/sell signals based on technical indicators.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data with technical indicators.
        
    Returns:
        pd.DataFrame: DataFrame with an additional column 'Signal' (1: Buy, -1: Sell, 0: Hold).
    """
    signals = np.zeros(len(df))
    
    for i in range(1, len(df)):
        # Buy Signal: RSI below 30 (oversold) and price crosses above lower Bollinger Band
        if df['RSI'].iloc[i] < 30 and df['Close'].iloc[i] > df['BB_Lower'].iloc[i]:
            signals[i] = 1
        
        # Sell Signal: RSI above 70 (overbought) and price crosses below upper Bollinger Band
        elif df['RSI'].iloc[i] > 70 and df['Close'].iloc[i] < df['BB_Upper'].iloc[i]:
            signals[i] = -1
        
    df['Signal'] = signals
    return df
