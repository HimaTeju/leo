import pandas as pd
from indicator_calculations import calculate_rsi, calculate_macd, calculate_bollinger_bands

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

        import pandas as pd
    from indicator_calculations import calculate_rsi, calculate_macd, calculate_bollinger_bands
    
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
    
        # def summary(self):
        #     # Summary of Buy and Sell signals
        #     buy_signals = self.signals['Buy'].sum()
        #     sell_signals = self.signals['Sell'].sum()
        #     return f"Buy Signals: {buy_signals}, Sell Signals: {sell_signals}"
    
        # def backtest(self, initial_balance=10000):
        #     balance = float(initial_balance)  # Ensuring balance is a scalar
        #     position = None
        #     peak_balance = float(initial_balance)  # Ensuring peak_balance is a scalar
        #     drawdown = float(initial_balance)  # Initialize drawdown with initial balance
        #     trade_history = []
        #     profitable_trades = 0
        #     total_trades = 0
    
        #     for i in range(1, len(self.signals)):
        #         # Buy logic
        #         if self.signals['Buy'].iloc[i] == 1 and position is None:
        #             position = balance / self.data['Close'].iloc[i]
        #             balance = 0
        #             trade_history.append(('Buy', self.data.index[i], self.data['Close'].iloc[i]))
    
        #         # Sell logic
        #         elif self.signals['Sell'].iloc[i] == 1 and position is not None:
        #             balance = position * self.data['Close'].iloc[i]
        #             position = None
        #             trade_history.append(('Sell', self.data.index[i], self.data['Close'].iloc[i]))
        #             total_trades += 1
    
        #     # Ensure both balance and peak_balance are scalar values
        #     peak_balance = max(float(peak_balance), float(balance.iloc[0]))
    
        #     # Track drawdown (lowest balance reached)
        #     drawdown = min(float(drawdown), float(balance.iloc[0]))
    
        #     # Check if balance is greater than initial balance
        #     if float(balance.iloc[0]) > initial_balance:  # Ensure balance is a float for comparison
        #         profitable_trades += 1
    
        #     # If position is still open, sell at the last price
        #     if position is not None:
        #         balance = position * self.data['Close'].iloc[-1]
        #         trade_history.append(('Sell', self.data.index[-1], self.data['Close'].iloc[-1]))
    
        #     # Calculate performance metrics
        #     total_return = (balance - initial_balance) / initial_balance * 100
        #     win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        #     drawdown_percent = (peak_balance - drawdown) / peak_balance * 100 if peak_balance > 0 else 0
    
        #     # Print performance metrics
        #     print(f"\nTotal Return: {total_return:.2f}%")
        #     print(f"Win Rate: {win_rate * 100:.2f}%")
        #     print(f"Maximum Drawdown: {drawdown_percent:.2f}%")
    
        #     return balance, trade_history