from backtester import TradingBacktester

try:
    # Initialize and run backtest
    backtester = TradingBacktester(
        symbol='AAPL',
        start_date='2022-01-01',
        end_date='2023-12-31',
        initial_capital=100000
    )

    # Run the backtest
    print("Starting backtesting process...")
    backtester.fetch_data()
    portfolio_values, positions, trades = backtester.backtest_strategy()

    # Print metrics
    metrics = backtester.calculate_risk_metrics(portfolio_values)
    print("\nBacktest Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2%}")

    # Show plots
    backtester.plot_results(portfolio_values, positions, trades)

except Exception as e:
    print(f"An error occurred: {str(e)}")
    raise e