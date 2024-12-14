import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_dataset(data, time_step):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def train_or_load_model(X_train, y_train, X_test, y_test, model_path, epochs=10, batch_size=32):
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    else:
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_test, y_test))
        model.save(model_path)
        return model


def predict_stock_prices(model, X_test, scaler):
    predicted_stock_price = model.predict(X_test)
    return scaler.inverse_transform(predicted_stock_price)


def calculate_metrics(y_test_actual, predicted_stock_price):
    rmse = math.sqrt(mean_squared_error(y_test_actual, predicted_stock_price))
    mae = mean_absolute_error(y_test_actual, predicted_stock_price)
    return rmse, mae


def predict_next_day_price(model, last_60_days, scaler):
    next_day_prediction = model.predict(last_60_days)
    return scaler.inverse_transform(next_day_prediction)[0][0]


def backtest_model(model, data, time_step, scaler):
    """
    Backtests the LSTM model and prints metrics to the terminal.

    Parameters:
    - model: Trained LSTM model.
    - data: Preprocessed and scaled stock data.
    - time_step: The number of time steps used for predictions.
    - scaler: The scaler used to normalize the data.

    Output:
    Prints backtesting metrics and actual vs predicted prices.
    """
    try:
        # Prepare the data for backtesting
        X_backtest, y_backtest = create_dataset(data, time_step)
        X_backtest = X_backtest.reshape(X_backtest.shape[0], X_backtest.shape[1], 1)

        # Predict using the model
        predictions = model.predict(X_backtest)
        predictions = scaler.inverse_transform(predictions)
        y_backtest_actual = scaler.inverse_transform(y_backtest.reshape(-1, 1))

        # Calculate metrics
        rmse, mae = calculate_metrics(y_backtest_actual, predictions)
        r2 = 1 - (sum((y_backtest_actual - predictions) ** 2) / sum((y_backtest_actual - y_backtest_actual.mean()) ** 2))

        # Print results to terminal
        print("\nBacktesting Results:")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared (R2): {r2[0]}")

        print("\nActual vs Predicted Prices:")
        for actual, predicted in zip(y_backtest_actual[:10], predictions[:10]):
            print(f"Actual: {actual[0]:.2f}, Predicted: {predicted[0]:.2f}")

    except Exception as e:
        print(f"Error during backtesting: {e}")