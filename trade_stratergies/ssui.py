import pandas as pd
import yfinance as yf
import numpy as np
import streamlit as st

# Fetch real stock data
def get_real_stock_data(ticker, start_date, end_date):
    """Fetch real stock data using yfinance."""
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data[['Open', 'High', 'Low', 'Close']]

# Clean column names if they are multi-indexed
def clean_column_names(df):
    """Clean column names if they are multi-indexed."""
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

class SignalGenerator:
    def __init__(self, stock_data):
        self.stock_data = clean_column_names(stock_data)  # Clean the columns here

    def generate_signals(self):
        # Calculate SMA (20 and 50 periods)
        self.stock_data['SMA_20'] = self.stock_data['Close'].rolling(window=20).mean()
        self.stock_data['SMA_50'] = self.stock_data['Close'].rolling(window=50).mean()

        # Calculate RSI (14-day)
        delta = self.stock_data['Close'].diff()
        gain = np.where(delta > 0, delta, 0).flatten()
        loss = np.where(delta < 0, -delta, 0).flatten()
        avg_gain = pd.Series(gain).rolling(window=14).mean()
        avg_loss = pd.Series(loss).rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.stock_data['RSI'] = 100 - (100 / (1 + rs))

        # Calculate MACD (12-26-9) - MACD line and Signal line
        ema_12 = self.stock_data['Close'].ewm(span=12, adjust=False).mean()  # 12-period EMA
        ema_26 = self.stock_data['Close'].ewm(span=26, adjust=False).mean()  # 26-period EMA
        self.stock_data['MACD'] = ema_12 - ema_26  # MACD Line
        self.stock_data['Signal_Line'] = self.stock_data['MACD'].ewm(span=9, adjust=False).mean()  # Signal Line

        # Generate Buy and Sell signals based on MACD crossover and SMA crossover
        self.stock_data['Buy'] = ((self.stock_data['MACD'] > self.stock_data['Signal_Line']).astype(int) + 
                                  (self.stock_data['SMA_20'] > self.stock_data['SMA_50']).astype(int) + 
                                  (self.stock_data['RSI'] < 30).astype(int)) >= 1  # Buy condition with RSI under 30
        self.stock_data['Sell'] = ((self.stock_data['MACD'] < self.stock_data['Signal_Line']).astype(int) + 
                                   (self.stock_data['SMA_20'] < self.stock_data['SMA_50']).astype(int) +
                                   (self.stock_data['RSI'] > 70).astype(int)) >= 1  # Sell condition with RSI over 70

        # Debug: Print out the first few rows of the signals
        st.write("Generated Buy and Sell signals:")
        st.dataframe(self.stock_data[['Buy', 'Sell', 'RSI', 'MACD', 'Signal_Line', 'SMA_20', 'SMA_50']].head(20))

        return self.stock_data

# Filter swing trades based on Buy and Sell signals
def filter_swing_trades(signals):
    """Filter for swing buy and sell signals."""
    swing_trades = signals[(signals['Buy'] == 1) | (signals['Sell'] == 1)]
    
    # Check for NaN values in key columns
    st.write("Checking for NaN values in the signals:")
    st.dataframe(swing_trades[['Buy', 'Sell', 'Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']].isna().sum())
    
    # Drop rows with NaN values in key columns
    swing_trades = swing_trades.dropna(subset=['Buy', 'Sell', 'Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line'])
    return swing_trades


# Streamlit Interface
def main():
    # Streamlit Input: Ticker and Date Range
    st.title('Stock Signal Generator')
    st.sidebar.header('Input Parameters')
    
    ticker = st.sidebar.text_input('Stock Ticker', 'RELIANCE.NS')
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('2023-12-31'))

    # Fetch stock data
    stock_data = get_real_stock_data(ticker, start_date, end_date)

    

    # Clean column names
    stock_data = clean_column_names(stock_data)

    # Generate signals
    signal_generator = SignalGenerator(stock_data)
    signals = signal_generator.generate_signals()

    # Display Buy/Sell signal counts
    st.write(f"Number of Buy signals: {signals['Buy'].sum()}")
    st.write(f"Number of Sell signals: {signals['Sell'].sum()}")

    # Display stock data with indicators
    # st.write("Stock Data with Indicators")
    # st.dataframe(signals[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']].tail())

    # Filter swing trades and display results
    swing_trades = filter_swing_trades(signals)
    
    # st.write(f"\nSwing Trades for {ticker}:")
    # st.dataframe(swing_trades[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 'Buy', 'Sell']])

    # Optionally, plot stock data and indicators
    st.write('### Stock Price with Indicators')
    st.line_chart(stock_data[['Close', 'SMA_20', 'SMA_50']])


if __name__ == "__main__":
    main()
