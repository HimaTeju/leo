import pandas as pd
from data_loader import load_data
from signal_generator import SignalGenerator
from backtest import backtest

# Example usage
if __name__ == "__main__":
    symbol = "ITC.NS"  # Example NSE symbol
    start_date = "2023-08-08"
    end_date = "2024-01-01"
    
    # Fetch stock data using the data_loader module
    stock_data = load_data(symbol, start_date=start_date, end_date=end_date)

    if stock_data is not None:
        print("\nStock Data (Head):")
        print(stock_data.head())

        # Create an instance of SignalGenerator and generate signals
        signal_generator = SignalGenerator(stock_data)
        signals = signal_generator.generate_signals()

        print("\nGenerated Signals (Buy=1, Sell=1):")
        print(signals.tail())

        # Print the summary of Buy and Sell signals
        print("\nBuy/Sell Signal Summary:")
        print(signal_generator.summary())

        # # Perform backtest with an initial balance of 10,000
        # final_balance, trade_history = backtest(signals, stock_data, initial_balance=10000)

        # # # Print the final balance and trade history
        # print("\nFinal Portfolio Value: ", final_balance)
        # print("\nTrade History:")
        # for trade in trade_history:
        #     print(trade)

        # # Performance metrics (already printed by backtest)
        # print(f"Final Balance: {final_balance}")
        # print(f"Total Return: {(final_balance - 10000) / 10000 * 100}%")
        # print(f"Total Trades: {len(trade_history)}")
