import pandas as pd
from buy_sell_signals.data_loader import load_data
from buy_sell_signals.signal_generator import SignalGenerator

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data using the data_loader module."""
    return load_data(symbol, start_date=start_date, end_date=end_date)

def display_stock_data(stock_data):
    """Display the first few rows of stock data."""
    print("\nStock Data (Head):")
    print(stock_data.head(20))

def generate_signals(stock_data):
    """Generate buy and sell signals using SignalGenerator."""
    signal_generator = SignalGenerator(stock_data)
    signals = signal_generator.generate_signals()

    print("\nGenerated Signals (Buy=1, Sell=1):")
    print(signals.tail(10))

    return signals, signal_generator

def filter_signals(signals):
    """Filter buy and sell signals from the generated signals."""
    buy_signals = signals[signals['Buy'] == 1]
    sell_signals = signals[signals['Sell'] == 1]

    print("Buy Signal Dates:")
    print(buy_signals[['Buy']])

    print("\nSell Signal Dates:")
    print(sell_signals[['Sell']])

    return buy_signals, sell_signals

def display_signal_summary(signal_generator):
    """Display the summary of buy and sell signals."""
    print("\nBuy/Sell Signal Summary:")
    print(signal_generator.summary())

def perform_backtest(signals, stock_data, initial_balance):
    """Perform backtest and display the results."""
    final_balance, trade_history = backtest(signals, stock_data, initial_balance)

    print("\nFinal Portfolio Value: ", final_balance)
    print("\nTrade History:")
    for trade in trade_history:
        print(trade)

    print(f"Final Balance: {final_balance}")
    print(f"Total Return: {(final_balance - initial_balance) / initial_balance * 100}%")
    print(f"Total Trades: {len(trade_history)}")

if __name__ == "__main__":
    # Parameters
    symbol = "INFY.NS"  # Example NSE symbol
    start_date = "2023-08-08"
    end_date = "2024-01-01"
    initial_balance = 10000

    # Fetch and display stock data
    stock_data = fetch_stock_data(symbol, start_date, end_date)
    if stock_data is not None:
        display_stock_data(stock_data)

        # Generate and display signals
        signals, signal_generator = generate_signals(stock_data)

        # Filter and display buy and sell signals
        filter_signals(signals)

        # Display signal summary
        display_signal_summary(signal_generator)