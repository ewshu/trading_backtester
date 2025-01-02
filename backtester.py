import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize


class TradingBacktester:
    def __init__(self, symbol, start_date, end_date, initial_capital=100000):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.commission = 0.001  # 0.1% commission per trade
        self.slippage = 0.0005  # 0.05% slippage per trade
        self.position = 0
        self.cash = initial_capital
        self.data = None
        self.ml_model = None

    def fetch_data(self):
        """Fetch historical data and calculate technical indicators"""
        print("Downloading data...")
        self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)

        print("Calculating technical indicators...")
        print("Converting data to correct format...")
        close_price = self.data['Close'].squeeze()

        print("Calculating SMA...")
        self.data['SMA_20'] = ta.trend.sma_indicator(close_price, window=20)
        self.data['SMA_50'] = ta.trend.sma_indicator(close_price, window=50)

        print("Calculating RSI...")
        self.data['RSI'] = ta.momentum.rsi(close_price, window=14)

        print("Calculating MACD...")
        self.data['MACD'] = ta.trend.macd_diff(close_price)

        print("Calculating Bollinger Bands...")
        indicator_bb = ta.volatility.BollingerBands(close=close_price, window=20, window_dev=2)
        self.data['BB_upper'] = indicator_bb.bollinger_hband()
        self.data['BB_middle'] = indicator_bb.bollinger_mavg()
        self.data['BB_lower'] = indicator_bb.bollinger_lband()

        # Drop NaN values
        self.data = self.data.dropna()
        print(f"Data shape after preparation: {self.data.shape}")

    def prepare_ml_features(self):
        """Prepare features for machine learning"""
        print("Preparing ML features...")
        features = pd.DataFrame(index=self.data.index)

        # Calculate SMA crossover
        features['SMA_cross'] = np.where(
            self.data['SMA_20'] > self.data['SMA_50'], 1, 0
        )

        # Add RSI
        features['RSI'] = self.data['RSI']

        # Add MACD
        features['MACD'] = self.data['MACD']

        print("Calculating BB position...")
        try:
            # Calculate BB position
            bb_range = self.data['BB_upper'] - self.data['BB_middle']
            bb_range = np.where(bb_range == 0, np.nan, bb_range)
            bb_position = (self.data['Close'] - self.data['BB_middle']) / bb_range
            features['BB_position'] = np.nan_to_num(bb_position, 0)
        except Exception as e:
            print(f"Error calculating BB position: {str(e)}")
            features['BB_position'] = 0

        # Create labels (1 for price increase, 0 for decrease)
        labels = np.where(self.data['Close'].shift(-1) > self.data['Close'], 1, 0)[:-1]

        # Remove last row of features to match labels
        features = features.iloc[:-1]

        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")

        return features, labels

    def train_ml_model(self):
        """Train Random Forest model for pattern recognition"""
        features, labels = self.prepare_ml_features()

        # Split data into training and validation sets
        train_size = int(len(features) * 0.8)
        X_train = features[:train_size]
        y_train = labels[:train_size]

        # Train model
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ml_model.fit(X_train, y_train)

    def calculate_transaction_costs(self, position_size):
        """Calculate transaction costs including commission and slippage"""
        commission_cost = float(abs(position_size) * self.commission)
        slippage_cost = float(abs(position_size) * self.slippage)
        return commission_cost + slippage_cost

    def backtest_strategy(self):
        """Run backtest with ML predictions and technical indicators"""
        print("Starting backtest...")
        if self.ml_model is None:
            self.train_ml_model()

        features = self.prepare_ml_features()[0]
        predictions = self.ml_model.predict_proba(features)

        # Initialize results tracking
        portfolio_value = []
        positions = []
        trades = []

        for i in range(len(self.data) - 1):
            current_price = float(self.data['Close'].iloc[i])

            # Trading logic combining ML and technical indicators
            ml_signal = predictions[i][1] > 0.6  # Probability threshold for long position
            rsi_signal = float(self.data['RSI'].iloc[i]) < 30  # Oversold condition
            sma_signal = float(self.data['SMA_20'].iloc[i]) > float(self.data['SMA_50'].iloc[i])

            # Combined signal
            buy_signal = ml_signal and (rsi_signal or sma_signal)
            sell_signal = not ml_signal and not (rsi_signal or sma_signal)

            # Execute trades
            if buy_signal and self.position == 0:
                position_size = self.cash * 0.95  # Use 95% of available cash
                shares = int(position_size / current_price)
                costs = self.calculate_transaction_costs(shares * current_price)

                if position_size - costs > 0:
                    self.position = shares
                    self.cash -= (shares * current_price + costs)
                    trades.append(('BUY', i, shares, current_price, costs))

            elif sell_signal and self.position > 0:
                costs = self.calculate_transaction_costs(self.position * current_price)
                self.cash += (self.position * current_price - costs)
                trades.append(('SELL', i, self.position, current_price, costs))
                self.position = 0

            # Track portfolio value
            portfolio_value.append(self.cash + self.position * current_price)
            positions.append(self.position)

        return portfolio_value, positions, trades

    def calculate_risk_metrics(self, portfolio_values):
        """Calculate various risk and performance metrics"""
        returns = pd.Series(portfolio_values).pct_change().dropna()

        metrics = {
            'Total Return': (portfolio_values[-1] - self.initial_capital) / self.initial_capital,
            'Annual Return': ((portfolio_values[-1] / self.initial_capital) ** (252 / len(portfolio_values)) - 1),
            'Sharpe Ratio': np.sqrt(252) * returns.mean() / returns.std(),
            'Max Drawdown': (pd.Series(portfolio_values).cummax() - portfolio_values).max() / pd.Series(
                portfolio_values).cummax().max(),
            'Volatility': returns.std() * np.sqrt(252),
        }

        return metrics

    def plot_results(self, portfolio_values, positions, trades):
        """Create professional-grade visualization of backtest results"""
        # Use a classic style
        plt.style.use('classic')
        # Set the figure background to white
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        # Plot 1: Portfolio Value
        dates = self.data.index[:-1]
        ax1.plot(dates, portfolio_values, label='Portfolio Value', linewidth=2)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Position Size
        ax2.plot(dates, positions, label='Position Size', color='orange', linewidth=2)
        ax2.set_title('Position Size Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of Shares')
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Trade Points on Price
        ax3.plot(self.data.index, self.data['Close'], label='Price', color='blue', alpha=0.7)

        # Add buy/sell markers
        for trade in trades:
            if trade[0] == 'BUY':
                ax3.scatter(self.data.index[trade[1]], self.data['Close'].iloc[trade[1]],
                            color='green', marker='^', s=100)
            else:
                ax3.scatter(self.data.index[trade[1]], self.data['Close'].iloc[trade[1]],
                            color='red', marker='v', s=100)

        ax3.set_title('Price and Trade Points')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Price ($)')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show()