import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf

# Streamlit App Configuration
st.set_page_config(page_title="Interactive Stock Dashboard", layout="wide")

# Sidebar: Navigation
st.sidebar.title("")
tab_home, tab_signals, tab_trends, tab_sentiment = st.tabs(["Home", "Buy/Sell Signals", "Predicted Trends", "Sentiment Scores"])

# Sidebar: Stock Input
st.sidebar.title("Stock Selection")
stock_name = st.sidebar.text_input("Enter Stock Symbol", value="reliance.ns", max_chars=20).upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# Validate Dates
if start_date >= end_date:
    st.error("Start date must be earlier than the end date. Please adjust the date range.")
    st.stop()

# Fetch Stock Data
@st.cache_data
def fetch_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data
    except Exception as e:
        return str(e)

# Fetch data for the selected stock
stock_data = fetch_stock_data(stock_name, start_date, end_date)

if stock_data is None or stock_data.empty:
    st.error(f"No data available for {stock_name}. Please check the stock symbol or date range.")
    st.stop()

# Ensure Close column is valid
if 'Close' not in stock_data or stock_data['Close'].dropna().empty:
    st.error(f"Stock data for {stock_name} is incomplete or invalid. Unable to proceed.")
    st.stop()

# Add Sentiment column with random scores for demo
stock_data['Sentiment'] = np.random.uniform(-1, 1, len(stock_data))
buy_signals = stock_data.iloc[::15].index
sell_signals = stock_data.iloc[::20].index

# Home Tab
with tab_home:
    st.title("📈 Stock Dashboard ")
    st.write(f"Displaying stock data for **{stock_name}** from {start_date} to {end_date}.")
    st.dataframe(stock_data)

# Buy/Sell Signals Tab
with tab_signals:
    st.title("💹 Buy/Sell Signals")
    buy_df = pd.DataFrame({"Date": list(buy_signals), "Signal": ["Buy"] * len(buy_signals)})
    sell_df = pd.DataFrame({"Date": list(sell_signals), "Signal": ["Sell"] * len(sell_signals)})
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Buy Signals")
        st.dataframe(buy_df)
    with col2:
        st.markdown("### Sell Signals")
        st.dataframe(sell_df)

# Predicted Trends Tab
with tab_trends:
    st.title("📊 Predicted Trends")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name="Stock Price"))
    fig_trend.update_layout(title=f"Predicted Stock Trend for {stock_name}", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_trend, use_container_width=True)

# Sentiment Scores Tab
with tab_sentiment:
    st.title("😊 Sentiment Scores")
    fig_sentiment = go.Figure()
    fig_sentiment.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data['Sentiment'],
            mode='lines+markers',
            name="Sentiment Score",
            marker=dict(color=["green" if s > 0 else "red" for s in stock_data['Sentiment']]),
        )
    )
    fig_sentiment.update_layout(title="Sentiment Analysis Over Time", xaxis_title="Date", yaxis_title="Sentiment Score")
    st.plotly_chart(fig_sentiment, use_container_width=True)
