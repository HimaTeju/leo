import streamlit as st
import pandas as pd
from buy_sell_signals.backend import fetch_stock_data, generate_signals, filter_signals, display_signal_summary, perform_backtest

def app():
    st.title("Stock Signal Generator & Backtesting")

    # Sidebar for user input
    st.sidebar.header("Buy and Sell signals")

    symbol = st.sidebar.text_input("Stock Symbol", "INFY.NS")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-08-08"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

    # Fetch Stock Data
    st.subheader(f"Fetching Data for {symbol} from {start_date} to {end_date}")
    stock_data = fetch_stock_data(symbol, start_date, end_date)
    
    if stock_data is not None:
        st.subheader("Stock Data")
        st.write(stock_data.head(20))

        # Generate Signals
        st.subheader("Generated Signals")
        signals, signal_generator = generate_signals(stock_data)
        st.write(signals)

        # Filter and display buy/sell signals
        st.subheader("Buy/Sell Signals")
        buy_signals, sell_signals = filter_signals(signals)
        
        # Display signal summary
        st.subheader("Signal Summary")
        st.write(signal_generator.summary())
    
if __name__ == "__main__":
    app()
