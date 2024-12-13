import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# Function to create the model architecture based on the saved model
def build_model(input_shape):
    model = Sequential()
    
    # LSTM layers and Dropout layers
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # Adjust dropout rate as needed

    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))  # Adjust dropout rate as needed

    model.add(Dense(1))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# Function to load the pre-trained model weights into the built model
def load_pretrained_model(model_path, input_shape):
    # Build the model architecture first
    model = build_model(input_shape)
    
    try:
        # Load the weights into the model
        model.load_weights(model_path)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
    
    return model

# Function to create dataset for LSTM model
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Function to preprocess data for training and prediction
def preprocess_data(df, column, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[[column]])

    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, y_train, X_test, y_test, scaler

# Function to predict stock prices using the loaded model
def predict_stock_prices(model, df, column, scaler, time_step=60):
    _, _, X_test, y_test, scaler = preprocess_data(df, column, time_step)
    predictions = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    return actual_prices, predicted_prices

# Example function to load data from CSV
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df
