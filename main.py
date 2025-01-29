### Step 1: Import Required Libraries

import backtrader as bt
import yfinance as yf
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from bayes_opt import BayesianOptimization

### Step 2: Download and Prepare Data

# Define the symbol and date range for data download
symbol = 'CL=F'  # Example: Crude Oil Futures (you can change it to any symbol)
start_date = '2017-01-01'  # Start date for historical data
end_date = '2023-01-01'  # End date for historical data

# Download the data from Yahoo Finance
data = yf.download(symbol, start=start_date, end=end_date)

# Flatten the MultiIndex columns and reset to simple column names
data.columns = [col[0] for col in data.columns]  # Flatten the MultiIndex

# Ensure the 'Date' column is datetime and set it as index
data['Date'] = pd.to_datetime(data.index)  # Ensure 'Date' is a datetime column
data.set_index('Date', inplace=True)  # Set 'Date' as index

### Step 3: Define the PCA-Based Strategy with Stop-Loss

class PCABasedAdvancedStrategyWithSL(bt.Strategy):
    def __init__(self, rsi_period=14, macd_fast_period=12, macd_slow_period=26, signal_window=14, stop_loss=0.05,
                 stochastic_period=14, williams_period=14, sma_fast=10, sma_slow=50, macro_trend_period=200,
                 macro_volume_period=200, atr_period=14, atr_multiplier=2):
        # Strategy initialization with parameters
        self.rsi_period = int(rsi_period)
        self.macd_fast_period = int(macd_fast_period)
        self.macd_slow_period = int(macd_slow_period)
        self.signal_window = int(signal_window)
        self.stop_loss = stop_loss
        self.stochastic_period = int(stochastic_period)
        self.williams_period = int(williams_period)
        self.sma_fast = int(sma_fast)
        self.sma_slow = int(sma_slow)
        self.macro_trend_period = int(macro_trend_period)
        self.macro_volume_period = int(macro_volume_period)
        self.atr_period = int(atr_period)
        self.atr_multiplier = atr_multiplier

        # Indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=self.rsi_period)
        self.macd = bt.indicators.MACD(self.data.close, period_me1=self.macd_fast_period, period_me2=self.macd_slow_period)
        self.stochastic = bt.indicators.Stochastic(self.data, period=self.stochastic_period)
        self.williams = bt.indicators.WilliamsR(self.data, period=self.williams_period)

        # Simple Moving Averages
        self.sma_fast = bt.indicators.SimpleMovingAverage(self.data.close, period=self.sma_fast)
        self.sma_slow = bt.indicators.SimpleMovingAverage(self.data.close, period=self.sma_slow)

        # Macro trend line (200-period SMA)
        self.macro_trend = bt.indicators.SimpleMovingAverage(self.data.close, period=self.macro_trend_period)

        # Macro Volume SMA
        self.macro_volume = bt.indicators.SimpleMovingAverage(self.data.volume, period=self.macro_volume_period)

        # ATR for Breakthrough Signals
        self.atr = bt.indicators.AverageTrueRange(self.data, period=self.atr_period)

        # Data collection for PCA
        self.data_frame = pd.DataFrame(columns=['RSI_signal', 'MACD_signal', 'Stochastic_signal',
                                                'WilliamsR_signal', 'SMA_Crossover_signal',
                                                'Macro_Trend_signal', 'Macro_Volume_signal', 'ATR_Breakthrough_signal'])
        self.pca_components = None

        # To track entry price and stop-loss
        self.entry_price = None

        # Statistics
        self.trades = []  # List to store trade results
        self.total_profit = 0  # Total profit/loss
        self.num_positive_trades = 0  # Number of positive trades
        self.num_negative_trades = 0  # Number of negative trades

    def next(self):
        if len(self.data) < max(self.macro_trend_period, self.macro_volume_period, self.atr_period):
            return

        # Generate binary signals based on indicators
        rsi_value = self.rsi[0]
        macd_value = self.macd.macd[0]
        macd_signal_value = self.macd.signal[0]
        stochastic_value = self.stochastic.percK[0]
        williams_value = self.williams[0]
        sma_fast_value = self.sma_fast[0]
        sma_slow_value = self.sma_slow[0]
        macro_trend_value = self.macro_trend[0]
        prev_macro_trend_value = self.macro_trend[-1]
        macro_volume_value = self.macro_volume[0]
        prev_macro_volume_value = self.macro_volume[-1]
        atr_value = self.atr[0]

        # Generate signals based on indicator thresholds
        signals = {
            'RSI_signal': 1 if rsi_value < 30 else (-1 if rsi_value > 70 else 0),
            'MACD_signal': 1 if macd_value > macd_signal_value else (-1 if macd_value < macd_signal_value else 0),
            'Stochastic_signal': 1 if stochastic_value < 20 else (-1 if stochastic_value > 80 else 0),
            'WilliamsR_signal': 1 if williams_value < -80 else (-1 if williams_value > -20 else 0),
            'SMA_Crossover_signal': 1 if sma_fast_value > sma_slow_value else (-1 if sma_fast_value < sma_slow_value else 0),
            'Macro_Trend_signal': 1 if macro_trend_value > prev_macro_trend_value else (-1 if macro_trend_value < prev_macro_trend_value else 0),
            'Macro_Volume_signal': 1 if macro_volume_value > prev_macro_volume_value else (-1 if macro_volume_value < prev_macro_volume_value else 0),
            'ATR_Breakthrough_signal': 1 if self.data.close[0] > (self.data.high[-1] + self.atr_multiplier * atr_value) else (-1 if self.data.close[0] < (self.data.low[-1] - self.atr_multiplier * atr_value) else 0)
        }

        # Update the data frame with binary signals
        self.data_frame = pd.concat([self.data_frame, pd.DataFrame([signals], columns=signals.keys())], ignore_index=True)

        # Ensure enough data points for PCA
        if len(self.data_frame) >= self.signal_window:
            # Use the last `signal_window` rows for PCA
            window_data = self.data_frame.iloc[-self.signal_window:]

            # Impute missing values (if any)
            imputer = SimpleImputer(strategy='mean')
            window_data_imputed = imputer.fit_transform(window_data)

            # Check if the imputed data is not empty
            if window_data_imputed.shape[0] > 0 and window_data_imputed.shape[1] > 0:
                # Apply PCA to imputed data
                pca = PCA(n_components=min(window_data_imputed.shape[1], 3))  # Use up to 3 components
                principal_components = pca.fit_transform(window_data_imputed)

                # Ensure principal components are available
                if principal_components.shape[1] > 0:
                    # Store the first principal component
                    self.pca_components = principal_components[:, 0]

                    # Generate signals based on PCA
                    if len(self.pca_components) > 1:
                        # Entry signals based on PCA trends
                        if self.pca_components[-1] > 0 and self.pca_components[-2] <= 0 and not self.position:
                            self.buy_signal()  # Buy signal (crossed from negative to positive)

                        elif self.pca_components[-1] < 0 and self.pca_components[-2] >= 0 and not self.position:
                            self.sell_signal()  # Sell signal (crossed from positive to negative)

                        # If we already have an open position, manage the opposite side of the trade
                        elif self.position:
                            if self.pca_components[-1] > 0 and self.pca_components[-2] <= 0 and self.position.size < 0:
                                self.close()  # Close short position
                                self.buy_signal()

                            elif self.pca_components[-1] < 0 and self.pca_components[-2] >= 0 and self.position.size > 0:
                                self.close()  # Close long position
                                self.sell_signal()

        # Check stop-loss logic if in a position
        if self.position:
            self.check_stop_loss()

    def buy_signal(self):
        """Handle Buy Signal."""
        self.buy(size=1)
        self.entry_price = self.data.close[0]

    def sell_signal(self):
        """Handle Sell Signal."""
        self.sell(size=1)
        self.entry_price = self.data.close[0]

    def check_stop_loss(self):
        """Close position if price hits stop-loss level."""
        if self.stop_loss != 0:
            stop_loss_price = self.entry_price * (1 - self.stop_loss if self.position.size > 0 else 1 + self.stop_loss)
            if (self.position.size > 0 and self.data.close[0] < stop_loss_price) or \
               (self.position.size < 0 and self.data.close[0] > stop_loss_price):
                self.close()  # Close the position

    def notify_trade(self, trade):
        """Called when a trade is completed."""
        if trade.isclosed:
            trade_result = trade.pnlcomm  # Profit/loss of the trade
            self.trades.append(trade_result)
            self.total_profit += trade_result

            if trade_result > 0:
                self.num_positive_trades += 1
            else:
                self.num_negative_trades += 1

### Step 4: Define the Optimization Function

def optimize_strategy(rsi_period, macd_fast_period, macd_slow_period, signal_window, stop_loss, stochastic_period, williams_period, sma_fast, sma_slow, macro_trend_period, macro_volume_period, atr_period, atr_multiplier):
    cerebro = bt.Cerebro()
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(PCABasedAdvancedStrategyWithSL, rsi_period=int(rsi_period), macd_fast_period=int(macd_fast_period), macd_slow_period=int(macd_slow_period), signal_window=int(signal_window), stop_loss=stop_loss, stochastic_period=int(stochastic_period), williams_period=int(williams_period), sma_fast=int(sma_fast), sma_slow=int(sma_slow), macro_trend_period=int(macro_trend_period), macro_volume_period=int(macro_volume_period), atr_period=int(atr_period), atr_multiplier=atr_multiplier)
    cerebro.broker.set_cash(100000)
    cerebro.run()

    # Return the total profit/loss as the optimization criterion
    return cerebro.run()[0].total_profit

### Step 5: Define the Parameter Space

pbounds = {
    'rsi_period': (10, 20),
    'macd_fast_period': (10, 14),
    'macd_slow_period': (20, 30),
    'signal_window': (10, 20),
    'stop_loss': (0.02, 0.1),
    'stochastic_period': (10, 20),
    'williams_period': (10, 20),
    'sma_fast': (5, 20),
    'sma_slow': (30, 100),
    'macro_trend_period': (100, 300),
    'macro_volume_period': (100, 300),
    'atr_period': (10, 20),
    'atr_multiplier': (1.5, 2.5)
}

### Step 6: Perform Bayesian Optimization

optimizer = BayesianOptimization(
    f=optimize_strategy,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=7,  # Initial random points
    n_iter=30,  # Number of optimization iterations
)

# Get the best parameters from optimization
best_params = optimizer.max["params"]
print(best_params)
# Set up Backtrader with the best parameters
cerebro = bt.Cerebro()
data_feed = bt.feeds.PandasData(dataname=data)
cerebro.adddata(data_feed)

# Add the strategy with the best parameters
cerebro.addstrategy(
    PCABasedAdvancedStrategyWithSL,
    rsi_period=int(best_params["rsi_period"]),
    macd_fast_period=int(best_params["macd_fast_period"]),
    macd_slow_period=int(best_params["macd_slow_period"]),
    signal_window=int(best_params["signal_window"]),
    stop_loss=best_params["stop_loss"],
    stochastic_period=int(best_params["stochastic_period"]),
    williams_period=int(best_params["williams_period"]),
    sma_fast=int(best_params["sma_fast"]),
    sma_slow=int(best_params["sma_slow"]),
    macro_trend_period=int(best_params["macro_trend_period"]),
    macro_volume_period=int(best_params["macro_volume_period"]),
    atr_period=int(best_params["atr_period"]),
    atr_multiplier=best_params["atr_multiplier"],
)

cerebro.broker.set_cash(1000)  # Set initial capital
cerebro.run()  # Run the best strategy

# Plot the Backtrader chart
cerebro.plot()