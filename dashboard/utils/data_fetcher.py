import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, period):
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker (str): The stock symbol (e.g., 'AAPL').
        period (str): The time period for the data (e.g., '1y', '6mo', '1mo').

    Returns:
        pd.DataFrame: Dataframe containing stock data with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    """
    stock_data = yf.download(ticker, period=period)
    stock_data.reset_index(inplace=True)
    return stock_data
