import requests
import json
import pandas as pd

API_KEY = 'JKR3HFT5WVA79TPN'
SYMBOL = 'TCS.BSE'  # For BSE-listed stock, or 'TCS.NS' for NSE
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={SYMBOL}&apikey={API_KEY}"

response = requests.get(url)
data = response.json()

# Extract the time series data
time_series = data.get('Time Series (Daily)', {})

# Convert the time series data to a pandas DataFrame
df = pd.DataFrame.from_dict(time_series, orient='index')

# Convert the index to datetime
df.index = pd.to_datetime(df.index)

# Sort the DataFrame by date
df = df.sort_index()

# Print the DataFrame
print(df.head())