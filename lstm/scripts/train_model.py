import numpy as np
import pandas as pd
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load stock symbols
stock_symbols = pd.read_csv('../data/NIFTY50_Stocks.csv')

# Download stock data
stocks_data = []
for symbol in stock_symbols['symbol']:
    stock_data = yf.download(symbol, start='2010-01-01', end='2023-12-31')
    stock_data['Symbol'] = symbol
    stocks_data.append(stock_data)

# Combine data for all stocks
full_stock_data = pd.concat(stocks_data, axis=0)

# Normalize the 'Close' prices
stock_close_data = full_stock_data[['Close', 'Symbol']]
scaler = MinMaxScaler(feature_range=(0, 1))
stock_close_data['Close'] = scaler.fit_transform(stock_close_data[['Close']])

# Prepare sequences for LSTM
def create_sequences(data, sequence_length=60):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(stock_close_data['Close'].values)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Save the model
model.save('models/lstm_model.h5')
