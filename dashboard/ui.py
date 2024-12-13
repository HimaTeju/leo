import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf

# Streamlit App Configuration
st.set_page_config(page_title="Interactive Stock Dashboard", layout="wide")

# Title
st.title("📈 Stock Dashboard")

# Input Section: Stock Name
st.sidebar.header("Stock Selection")
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
        return  str(e)

# Fetch data for the selected stock
stock_data  = fetch_stock_data(stock_name, start_date, end_date)

if stock_data is None or stock_data.empty:
    st.error(f"No data available for {stock_name}. Please check the stock symbol or date range.")
else:
    st.success(f"Data loaded successfully for {stock_name}")

    # Ensure Close column is valid
    if 'Close' not in stock_data or stock_data['Close'].dropna().empty:
        st.error(f"Stock data for {stock_name} is incomplete or invalid. Unable to proceed.")
        st.stop()

    # Add Sentiment column with random scores for demo
    stock_data['Sentiment'] = np.random.uniform(-1, 1, len(stock_data))
    buy_signals = stock_data.iloc[::15].index
    sell_signals = stock_data.iloc[::20].index

   # Section 1: Buy/Sell Signals
st.subheader("Buy/Sell Signals")
with st.expander("View Signals"):
    # Create separate dataframes for Buy and Sell signals
    buy_df = pd.DataFrame({"Date": list(buy_signals), "Signal": ["Buy"] * len(buy_signals)})
    sell_df = pd.DataFrame({"Date": list(sell_signals), "Signal": ["Sell"] * len(sell_signals)})

    # Display Buy and Sell signals in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Buy Signals")
        st.dataframe(buy_df)

    with col2:
        st.markdown("### Sell Signals")
        st.dataframe(sell_df)

    # Section 2: Predicted Trends
    st.subheader("Predicted Trends")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name="Stock Price"))
    fig_trend.update_layout(title=f"Predicted Stock Trend for {stock_name}", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_trend, use_container_width=True)

    # Section 3: Sentiment Scores
    st.subheader("Sentiment Scores")
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
    fig_sentiment.update_layout(
        title="Sentiment Analysis Over Time", xaxis_title="Date", yaxis_title="Sentiment Score"
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # Section 4: Support/Resistance Levels
    st.subheader("Support and Resistance Levels")
    support_level = stock_data['Close'].dropna().min() + 10  # Ensure min value is calculated without NaN
    resistance_level = stock_data['Close'].dropna().max() - 10  # Ensure max value is calculated without NaN

    fig_sr = go.Figure()
    fig_sr.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name="Price", line=dict(color="blue")))
    fig_sr.add_hline(y=float(support_level), line_dash="dash", line_color="green", annotation_text="Support Level")
    fig_sr.add_hline(y=float(resistance_level), line_dash="dash", line_color="red", annotation_text="Resistance Level")
    fig_sr.update_layout(title="Support and Resistance Levels", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_sr, use_container_width=True)
