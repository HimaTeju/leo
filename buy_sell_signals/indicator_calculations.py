import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Fetch data from Yahoo Finance
def fetch_data(symbol, start_date='2023-08-08', end_date='2024-01-01'):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Calculate RSI
def calculate_rsi(data, period=14):
    # Ensure 'Close' is a Series (1-dimensional)
    close_prices = data['Close'].squeeze()  # This ensures it's a 1-dimensional array/series
    rsi = RSIIndicator(close_prices, window=period).rsi()
    return rsi

# Calculate MACD
def calculate_macd(data):
    # Ensure 'Close' is a Series (1-dimensional)
    close_prices = data['Close'].squeeze()  # This ensures it's a 1-dimensional array/series

    # Calculate MACD using the ta library
    macd = MACD(close_prices)

    # Calculate MACD line and signal line
    macd_line = macd.macd()  # MACD line (difference between fast and slow EMAs)
    signal_line = macd.macd_signal()  # Signal line (EMA of the MACD line)

    return macd_line, signal_line


# Generate buy and sell signals
def generate_signals(data):
    signals = pd.DataFrame(index=data.index)

    # Calculate RSI and MACD
    signals['RSI'] = calculate_rsi(data)
    macd_line, signal_line = calculate_macd(data)
    
    # Add MACD and Signal to the signals dataframe
    signals['MACD'] = macd_line
    signals['Signal'] = signal_line

    # Buy and Sell signals based on RSI and MACD
    signals['Buy'] = (signals['RSI'] < 40) | (signals['MACD'] > signals['Signal'])
    signals['Sell'] = (signals['RSI'] > 60) & (signals['MACD'] < signals['Signal'])
    
    # Convert boolean to integer (1 for True, 0 for False)
    signals['Buy'] = signals['Buy'].astype(int)
    signals['Sell'] = signals['Sell'].astype(int)

    return signals

# Example usage
if __name__ == "__main__":
    symbol = "HDFCBANK.NS"  # Example NSE symbol
    start_date = "2023-08-08"
    end_date = "2024-01-01"
    
    # Fetch stock data
    stock_data = fetch_data(symbol, start_date=start_date, end_date=end_date)

    if stock_data is not None:
        print("\nStock Data (Head):")
        print(stock_data.head())

        # Generate signals
        signals = generate_signals(stock_data)
        
        print("\nGenerated Signals (Buy=1, Sell=1):")
        print(signals.tail())

        # Summary of Buy and Sell signals
        print("\nBuy/Sell Signal Summary:")
        print(f"Buy Signals: {signals['Buy'].sum()}")
        print(f"Sell Signals: {signals['Sell'].sum()}")
