import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

# Function to prepare the data for LSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])  # Use previous 'time_step' prices to predict the next
        y.append(data[i, 0])  # The next day's price (target)
    return np.array(X), np.array(y)

# Function to load and preprocess the stock data
def load_data(ticker, start="2010-01-01", end="2023-01-01"):
    data = yf.download(ticker, start=start, end=end)
    close_price = data['Close'].values
    close_price = close_price.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_price)
    
    return scaled_data, scaler, data

# Function to build and train the LSTM model
def build_train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))  # Output layer to predict the next day's closing price

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    return model

# Streamlit app
def main():
    st.title("Stock Price Prediction with LSTM")

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA):", "AAPL")
    
    if ticker:
        # Load and preprocess data
        scaled_data, scaler, data = load_data(ticker)
        
        # Prepare the dataset
        X, y = create_dataset(scaled_data)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Build and train the model
        model = build_train_model(X_train, y_train)
        
        # Predict stock prices
        predicted_stock_price = model.predict(X_test)
        
        # Invert scaling to get actual stock prices
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate RMSE
        rmse = math.sqrt(mean_squared_error(y_test_actual, predicted_stock_price))
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_actual, color='red', label='Actual Stock Price')
        plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
        plt.title(f'Stock Price Prediction for {ticker} using LSTM')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        
        st.pyplot(plt)

# Run the Streamlit app
if __name__ == "__main__":
    main()
