import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lstm_model import (
    create_dataset,
    train_or_load_model,
    predict_stock_prices,
    calculate_metrics,
    predict_next_day_price,
)
import os
from datetime import datetime, timedelta

st.title("Stock Price Prediction using LSTM")

# Dynamic date calculation
end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365 * 10)  # 10 years ago

# Input ticker
st.sidebar.header("Stock Selection")
stock_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TCS.NS, RELIANCE.BO):", "TCS.NS")

# Date inputs with defaults
start_date = st.sidebar.date_input("Start Date:", value=start_date_default)
end_date = st.sidebar.date_input("End Date:", value=end_date_default)

time_step = st.sidebar.slider("Time Steps (Days):", min_value=10, max_value=100, value=60, step=10)
epochs = st.sidebar.slider("Epochs:", min_value=1, max_value=50, value=10)
batch_size = st.sidebar.slider("Batch Size:", min_value=16, max_value=128, value=32)

if st.sidebar.button("Predict Stock Prices"):
    model_dir = f"models/{stock_ticker}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/model.h5"

    st.write(f"Fetching data for {stock_ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(stock_ticker, start=start_date, end=end_date)
        st.write("### Fetched Data")
        st.write(data)

        if data.empty:
            st.error("No data found for the entered stock ticker. Please try again.")
        else:
            close_price = data['Close'].values.reshape(-1, 1)

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_price)

            X, y = create_dataset(scaled_data, time_step)
            X = X.reshape(X.shape[0], X.shape[1], 1)

            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            if os.path.exists(model_path):
                st.write(f"Model found at `{model_path}`. Loading the existing model...")
                model = train_or_load_model(X_train, y_train, X_test, y_test, model_path, epochs, batch_size)
                st.success("Successfully loaded the existing model.")
            else:
                st.write("No pre-existing model found. Creating and training a new model...")
                model = train_or_load_model(X_train, y_train, X_test, y_test, model_path, epochs, batch_size)
                st.success("Successfully created and trained a new model.")

            predicted_stock_price = predict_stock_prices(model, X_test, scaler)
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

            rmse, mae = calculate_metrics(y_test_actual, predicted_stock_price)
            st.write(f"Root Mean Squared Error (RMSE): {rmse}")
            st.write(f"Mean Absolute Error (MAE): {mae}")

            last_60_days = scaled_data[-time_step:].reshape(1, -1, 1)
            next_day_price = predict_next_day_price(model, last_60_days, scaler)
            st.write(f"Predicted Next Day Price: {next_day_price:.2f}")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(y_test_actual, color='red', label='Actual Stock Price')
            ax.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
            ax.set_title(f'Stock Price Prediction for {stock_ticker}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Stock Price')
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
