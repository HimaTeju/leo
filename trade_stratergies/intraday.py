import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Streamlit app title
st.title("Intraday Stock Analysis")

# Sidebar for inputs
st.sidebar.header("Inputs")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS):", "RELIANCE.NS")
analysis_date = st.sidebar.date_input("Select a Date for Analysis:", datetime.now().date())
time_frame = st.sidebar.selectbox("Select Time Frame:", ["1h", "30m", "15m", "5m"], index=0)

# Fetch intraday data
def fetch_intraday_data(symbol, date, interval):
    start_date = datetime.combine(date, datetime.min.time())
    end_date = start_date + timedelta(days=2)
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    return data

# Technical indicators
def calculate_indicators(data):
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = calculate_rsi(data['Close'], 14)
    return data

def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Fetch and process data
try:
    st.write(f"Fetching data for {stock_symbol} starting from {analysis_date} with a time frame of {time_frame}...")
    data = fetch_intraday_data(stock_symbol, analysis_date, time_frame)
    data = calculate_indicators(data)

    # Display raw data
    st.write("### Raw Data")
    st.dataframe(data.tail())

    # Plot candlestick chart with indicators
    st.write("### Candlestick Chart with Indicators")
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlesticks',
                                 increasing_line_color='green',
                                 decreasing_line_color='red'))

    # SMA20
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], mode='lines', name='SMA20', line=dict(color='blue')))

    # Layout
    fig.update_layout(title=f"Intraday Analysis for {stock_symbol}",
                      xaxis_title="Time",
                      yaxis_title="Price",
                      template="plotly_dark",
                      xaxis_rangeslider_visible=False)

    st.plotly_chart(fig)

    # RSI Visualization
    st.write("### Relative Strength Index (RSI)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='orange')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.update_layout(title="RSI Chart", xaxis_title="Time", yaxis_title="RSI", template="plotly_dark")
    st.plotly_chart(fig_rsi)

    # Highlight buy/sell signals based on RSI
    st.write("### RSI Signals")
    buy_signals = data[data['RSI'] < 30]
    sell_signals = data[data['RSI'] > 70]

    st.write("*Buy Signals:*")
    st.dataframe(buy_signals[['Close', 'RSI']])

    st.write("*Sell Signals:*")
    st.dataframe(sell_signals[['Close', 'RSI']])

except Exception as e:
    st.error(f"An error occurred: {e}")