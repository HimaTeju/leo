def backtest(signals, data, initial_balance=10000):
    balance = float(initial_balance)
    position = None
    peak_balance = float(initial_balance)
    drawdown = 0
    trade_history = []
    profitable_trades = 0
    total_trades = 0
    
    for i in range(1, len(signals)):
        if signals['Buy'].iloc[i] == 1 and position is None:
            position = balance / data['Close'].iloc[i]
            balance = 0
            trade_history.append(('Buy', data.index[i], data['Close'].iloc[i], balance))
            total_trades += 1
        
        elif signals['Sell'].iloc[i] == 1 and position is not None:
            balance = position * data['Close'].iloc[i]
            position = None
            trade_history.append(('Sell', data.index[i], data['Close'].iloc[i], balance))
            total_trades += 1
            if balance > peak_balance:
                peak_balance = balance
            drawdown = peak_balance - balance if balance < peak_balance else 0
            if balance > initial_balance:
                profitable_trades += 1
    
    if position is not None:
        balance = position * data['Close'].iloc[-1]
        trade_history.append(('Sell', data.index[-1], data['Close'].iloc[-1], balance))
    
    total_return = (balance - initial_balance) / initial_balance * 100
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    drawdown_percent = (peak_balance - drawdown) / peak_balance * 100 if peak_balance > 0 else 0
    
    print(f"\nTotal Return: {total_return:.2f}%")
    print(f"Win Rate: {win_rate * 100:.2f}%")
    print(f"Maximum Drawdown: {drawdown_percent:.2f}%")
    
    return balance, trade_history