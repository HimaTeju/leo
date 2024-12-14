import pandas as pd
from buy_sell_signals.indicator_calculations import calculate_rsi, calculate_macd, calculate_bollinger_bands

class SignalGenerator:
    def __init__(self, data):
        self.data = data
        self.signals = pd.DataFrame(index=data.index)

    def generate_signals(self):
        self.signals['RSI'] = calculate_rsi(self.data)
        macd_data = calculate_macd(self.data)
        if macd_data is not None:
            macd_line, signal_line = macd_data
            self.signals['MACD'] = macd_line
            self.signals['Signal'] = signal_line

        bollinger_data = calculate_bollinger_bands(self.data)
        if bollinger_data is not None:
            upper_band, middle_band, lower_band = bollinger_data
            self.signals['Upper_Band'] = upper_band
            self.signals['Middle_Band'] = middle_band
            self.signals['Lower_Band'] = lower_band

        # Reindex signals to match data
        self.signals = self.signals.reindex(self.data.index)

        # Extract Close prices
        close_prices = self.data['Close'].squeeze()

        # Explicit alignment
        close_prices, lower_band = close_prices.align(self.signals['Lower_Band'], join='inner')
        close_prices, upper_band = close_prices.align(self.signals['Upper_Band'], join='inner')

        # Debugging outputs
        print("Aligned Close Prices:\n", close_prices.head())
        print("Aligned Lower Band:\n", lower_band.head())
        print("Aligned Upper Band:\n", upper_band.head())

        # Generate Buy and Sell signals
        self.signals['Buy'] = (
            (self.signals['RSI'] < 30).astype(int) +
            (self.signals['MACD'] > self.signals['Signal']).astype(int) +
            (close_prices <= lower_band).astype(int)
        ) >= 2

        self.signals['Sell'] = (
            (self.signals['RSI'] > 70).astype(int) +
            (self.signals['MACD'] < self.signals['Signal']).astype(int) +
            (close_prices >= upper_band).astype(int)
        ) >= 2

        # Convert to integers
        self.signals['Buy'] = self.signals['Buy'].astype(int)
        self.signals['Sell'] = self.signals['Sell'].astype(int)

        return self.signals

    def summary(self):
        # Summary of Buy and Sell signals
        buy_signals = self.signals['Buy'].sum()
        sell_signals = self.signals['Sell'].sum()
        return f"Buy Signals: {buy_signals}, Sell Signals: {sell_signals}"

    
    class SignalGenerator:
        def __init__(self, data):
            self.data = data
            self.signals = pd.DataFrame(index=data.index)
    
        def generate_signals(self):
            self.signals['RSI'] = calculate_rsi(self.data)
            macd_data = calculate_macd(self.data)
            if macd_data is not None:
                macd_line, signal_line = macd_data
                self.signals['MACD'] = macd_line
                self.signals['Signal'] = signal_line
    
            bollinger_data = calculate_bollinger_bands(self.data)
            if bollinger_data is not None:
                upper_band, middle_band, lower_band = bollinger_data
                self.signals['Upper_Band'] = upper_band
                self.signals['Middle_Band'] = middle_band
                self.signals['Lower_Band'] = lower_band
    
            # Reindex signals to match data
            self.signals = self.signals.reindex(self.data.index)
    
            # Extract Close prices
            close_prices = self.data['Close'].squeeze()
    
            # Explicit alignment
            close_prices, lower_band = close_prices.align(self.signals['Lower_Band'], join='inner')
            close_prices, upper_band = close_prices.align(self.signals['Upper_Band'], join='inner')
    
            # Debugging outputs
            print("Aligned Close Prices:\n", close_prices.head())
            print("Aligned Lower Band:\n", lower_band.head())
            print("Aligned Upper Band:\n", upper_band.head())
    
            # Generate Buy and Sell signals
            self.signals['Buy'] = (
                (self.signals['RSI'] < 30).astype(int) +
                (self.signals['MACD'] > self.signals['Signal']).astype(int) +
                (close_prices <= lower_band).astype(int)
            ) >= 2
    
            self.signals['Sell'] = (
                (self.signals['RSI'] > 70).astype(int) +
                (self.signals['MACD'] < self.signals['Signal']).astype(int) +
                (close_prices >= upper_band).astype(int)
            ) >= 2
    
            # Convert to integers
            self.signals['Buy'] = self.signals['Buy'].astype(int)
            self.signals['Sell'] = self.signals['Sell'].astype(int)
    
            return self.signals
    
       