import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data using yfinance."""
    return yf.download(ticker, start=start_date, end=end_date)


def preprocess_data(df):
    """Preprocess stock data for prediction."""
    df['Date'] = pd.to_datetime(df.index)
    df['Date'] = df['Date'].map(lambda x: x.toordinal())
    X = df['Date'].values.reshape(-1, 1)
    y = df['Close'].values
    return X, y


def train_model(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return predictions and metrics."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return y_pred, mse


def plot_predictions(df, y_test, y_pred):
    """Plot actual vs. predicted stock prices."""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-len(y_test):], y_test, label='Actual', color='blue')
    plt.plot(df.index[-len(y_pred):], y_pred, label='Predicted', color='red')
    plt.title("Prediction vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    return plt


def predict_next_day(model, last_date):
    """Predict the stock price for the next day."""
    next_day = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    next_day_ordinal = np.array([next_day.toordinal()]).reshape(-1, 1)
    predicted_price = model.predict(next_day_ordinal)
    return next_day.date(), predicted_price[0].item()
