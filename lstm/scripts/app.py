import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from predict_stock import load_pretrained_model, preprocess_data, predict_stock_prices

st.title("Stock Price Prediction using LSTM")

# Sidebar inputs for stock selection and model configuration
st.sidebar.header("Stock Selection")
stock_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TCS.NS):", "TCS.NS")

start_date = st.sidebar.date_input("Start Date:", datetime.today() - timedelta(days=365 * 5))
end_date = st.sidebar.date_input("End Date:", datetime.today())

time_step = st.sidebar.slider("Time Steps (Days):", min_value=10, max_value=100, value=60, step=10)

if st.sidebar.button("Predict Stock Prices"):
    try:
        st.write(f"Fetching data for {stock_ticker} from {start_date} to {end_date}...")
        data = yf.download(stock_ticker, start=start_date, end=end_date)

        if data.empty:
            st.error("No data found for the entered stock ticker. Please try again.")
        else:
            st.write("### Fetched Data")
            st.write(data)

            # Preprocess data
            close_prices = data['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)

            # Ensure there are no NaN values in the scaled data
            if np.any(np.isnan(scaled_data)):
                st.error("Data contains NaN values after scaling.")
                

            # Load the model
            model_path = "models/lstm_model.h5"
            input_shape = (time_step, 1)
            model = load_pretrained_model(model_path, input_shape)

            # Ensure the model is loaded correctly
            if model is None:
                st.error("Failed to load the model.")
                

            # Predict stock prices
            actual_prices, predicted_prices = predict_stock_prices(model, data, 'Close', scaler, time_step)

            if np.any(np.isnan(predicted_prices)) or np.any(np.isnan(actual_prices)):
                st.error("Prediction contains NaN values.")
                

            # Display metrics and predictions
            st.write("### Prediction Metrics")
            rmse = np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

            st.write("### Predicted Next Day Price")
            last_60_days = scaled_data[-time_step:].reshape(1, -1, 1)
            next_day_price = model.predict(last_60_days)
            next_day_price = scaler.inverse_transform(next_day_price)
            
            # Ensure the next day prediction isn't NaN
            if np.any(np.isnan(next_day_price)):
                st.error("Predicted Next Day Price is NaN.")
                

            st.write(f"Predicted Next Day Price: {next_day_price[0][0]:.2f}")

            # Plot predictions
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(actual_prices, color='red', label='Actual Stock Price')
            ax.plot(predicted_prices, color='blue', label='Predicted Stock Price')
            ax.set_title(f'Stock Price Prediction for {stock_ticker}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Stock Price')
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
