import yfinance as yf
import talib as ta
import pandas as pd
import matplotlib.pyplot as plt

# Fetch stock data for TCS from NSE (you can change the ticker symbol)
ticker = 'TCS.NS'
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# Convert 'Close' column to a numpy array and flatten it to 1D
close_price = data['Close'].to_numpy().flatten()

# Debugging: Check the shape of close_price
print("Shape of close_price:", close_price.shape)

# Ensure the array is 1D
if close_price.ndim != 1:
    raise ValueError("close_price should be a 1D array")

# Calculate RSI, MACD, and Bollinger Bands
data['RSI'] = ta.RSI(close_price, timeperiod=14)
data['MACD'], data['MACD_signal'], _ = ta.MACD(close_price, fastperiod=12, slowperiod=26, signalperiod=9)
data['Upper_BB'], data['Middle_BB'], data['Lower_BB'] = ta.BBANDS(close_price, timeperiod=20)

# Display the latest values in a readable format
latest_data = data.iloc[-1]

print(f"Stock Ticker: {ticker}")
print(f"Latest Closing Price: ₹{latest_data['Close']}")
print(f"RSI: {latest_data['RSI']}")
print(f"MACD: {latest_data['MACD']}")
print(f"MACD Signal: {latest_data['MACD_signal']}")
print(f"Bollinger Bands:")
print(f"  Upper Band: ₹{latest_data['Upper_BB']}")
print(f"  Middle Band: ₹{latest_data['Middle_BB']}")
print(f"  Lower Band: ₹{latest_data['Lower_BB']}")

# Plotting the closing price and Bollinger Bands
plt.figure(figsize=(10,6))

# Plot the closing price
plt.plot(data['Close'], label='Close Price', color='blue')

# Plot the Bollinger Bands
plt.plot(data['Upper_BB'], label='Upper Bollinger Band', linestyle='--', color='red')
plt.plot(data['Middle_BB'], label='Middle Bollinger Band', linestyle='--', color='green')
plt.plot(data['Lower_BB'], label='Lower Bollinger Band', linestyle='--', color='red')

plt.title(f"{ticker} Stock Price with Bollinger Bands")
plt.legend(loc='upper left')
plt.xlabel('Date')
plt.ylabel('Price (₹)')
plt.grid(True)
plt.show()

