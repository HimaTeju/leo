import yfinance as yf
import talib as ta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Fetch stock data
stock_ticker = 'TCS.NS'  # Replace with the stock you want
data = yf.download(stock_ticker, start='2020-01-01', end='2024-01-01')

# Calculate technical indicators
close_price = data['Close'].to_numpy().squeeze()  # Ensure it's a 1D array

# Ensure that the Close price is 1D
assert close_price.ndim == 1, "close_price should be a 1D array"

# Calculate RSI
data['RSI'] = ta.RSI(close_price, timeperiod=14)

# Calculate other indicators
data['MACD'], data['MACD_signal'], _ = ta.MACD(close_price, fastperiod=12, slowperiod=26, signalperiod=9)
data['50_MA'] = data['Close'].rolling(window=50).mean()
data['200_MA'] = data['Close'].rolling(window=200).mean()
data['Price Change'] = data['Close'].diff()
# Calculate Bollinger Bands
data['Upper_Band'], data['Middle_Band'], data['Lower_Band'] = ta.BBANDS(close_price, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

# Remove NaN values
data.dropna(inplace=True)

# Define target variable: 1 if price goes up, 0 if price goes down
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Prepare features and target
features = ['RSI', 'MACD', '50_MA', '200_MA', 'Price Change', 'Upper_Band', 'Lower_Band']
target = 'Target'

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot feature importance
feature_importance = model.feature_importances_
plt.barh(features, feature_importance)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()

# Predict stock movement for the next day (last row)
latest_data = data.tail(1)[features]
prediction = model.predict(latest_data)

if prediction == 1:
    print("The stock is predicted to go UP tomorrow.")
else:
    print("The stock is predicted to go DOWN tomorrow.")
