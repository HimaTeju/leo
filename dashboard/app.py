import streamlit as st
import pandas as pd
from utils.data_fetcher import fetch_stock_data
from utils.indicators import compute_technical_indicators
from utils.signals import generate_signals
from plotly.graph_objs import Candlestick, Scatter

# Streamlit UI for User Input
st.title("Stock Trading Strategies Dashboard")

ticker = st.text_input("Enter Stock Ticker", "AAPL")
period = st.selectbox("Select Time Period", ["1y", "6mo", "3mo", "1mo"])

if ticker:
    stock_data = fetch_stock_data(ticker, period)
    stock_data = compute_technical_indicators(stock_data)
    stock_data = generate_signals(stock_data)
    
    st.write("Latest Stock Data:")
    st.write(stock_data.tail())

    # Signal Output
    latest_signal = stock_data['Signal'].iloc[-1]
    if latest_signal == 1:
        st.write(f"**Buy Signal** for {ticker}")
    elif latest_signal == -1:
        st.write(f"**Sell Signal** for {ticker}")
    else:
        st.write(f"**Hold Signal** for {ticker}")

    # Plot the Stock Chart
    fig = plot_stock_chart(stock_data)
    st.plotly_chart(fig)
