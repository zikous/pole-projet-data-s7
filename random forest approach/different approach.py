import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class MarketDataHandler:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = None

    def fetch_data(self, start_date, end_date):
        """Fetch market data and create features"""
        try:
            self.data = yf.download(self.symbol, start=start_date, end=end_date)
            if self.data.empty:
                raise ValueError(f"No data available for {self.symbol}")
            print(f"Fetched {len(self.data)} days of data")  # Debug print
            return self.data
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")

    def calculate_features(self, data_slice):
        """Calculate features using only available data up to current point"""
        features = pd.DataFrame(index=data_slice.index)

        # Basic features without lookback
        features["Open-Close"] = (data_slice.Open - data_slice.Close) / data_slice.Open
        features["High-Low"] = (data_slice.High - data_slice.Low) / data_slice.Low

        # Momentum indicators
        features["price_momentum"] = data_slice["Close"].pct_change()
        features["volume_momentum"] = data_slice["Volume"].pct_change()

        # Moving averages with shorter windows
        features["MA5"] = data_slice["Close"].rolling(window=5).mean()
        features["MA10"] = data_slice["Close"].rolling(window=10).mean()
        features["MA_ratio"] = features["MA5"] / features["MA10"]

        # Volatility (shorter window)
        features["volatility"] = data_slice["Close"].rolling(5).std()

        # Price relative to moving averages
        features["price_ma5_ratio"] = data_slice["Close"] / features["MA5"]

        return features.fillna(method="ffill").fillna(0)  # Fill NaN values

    def get_feature_names(self):
        """Return list of feature columns for training"""
        return [
            "Open-Close",
            "High-Low",
            "price_momentum",
            "volume_momentum",
            "MA_ratio",
            "volatility",
            "price_ma5_ratio",
        ]


class TradingModel:
    def __init__(self, n_estimators=100, max_depth=3, min_samples_split=5):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
        )
        self.feature_names = None

    def prepare_data(self, features, prices):
        """Prepare data for training"""
        X = features[self.feature_names]

        # Target: 1 if price went up by at least 0.1%, -1 if it went down by at least 0.1%
        returns = prices["Close"].pct_change().shift(-1)
        y = np.where(returns > 0.001, 1, np.where(returns < -0.001, -1, 0))[:-1]

        return X[:-1], y

    def train_and_predict(self, X_train, y_train, X_predict):
        """Train model and make prediction with debugging"""
        if len(X_train) < 10:  # Require minimum amount of training data
            print(f"Insufficient training data: {len(X_train)} samples")
            return 0

        try:
            # Print shapes for debugging
            print(f"Training data shape: {X_train.shape}")
            print(f"Training labels shape: {y_train.shape}")
            print(f"Prediction data shape: {X_predict.shape}")

            self.model.fit(X_train, y_train)
            prediction = self.model.predict(X_predict)[0]
            print(f"Model prediction: {prediction}")
            return prediction

        except Exception as e:
            print(f"Model error: {str(e)}")
            return 0


class DayTrader:
    def __init__(
        self, symbol, initial_capital=100000, position_size=0.95, stop_loss=0.02
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.trades = []
        self.daily_values = []

        self.data_handler = MarketDataHandler(symbol)
        self.model = TradingModel()

    def _execute_trade(self, date, prediction, prices):
        """Execute trade based on prediction"""
        print(f"Attempting trade on {date} with prediction {prediction}")  # Debug print

        if prediction == 0:
            print("No trade: Neutral prediction")
            return

        try:
            current_price = prices["Open"]
            trade_amount = self.current_capital * self.position_size
            shares = int(trade_amount / current_price)

            if shares == 0:
                print("No trade: Insufficient capital for one share")
                return

            entry_price = current_price
            exit_price = prices["Close"]
            stop_loss_price = entry_price * (
                1 - self.stop_loss if prediction == 1 else 1 + self.stop_loss
            )

            # Check if stop loss was hit
            if prediction == 1:  # Long position
                if prices["Low"] <= stop_loss_price:
                    exit_price = stop_loss_price
                trade_pnl = (exit_price - entry_price) * shares
            else:  # Short position
                if prices["High"] >= stop_loss_price:
                    exit_price = stop_loss_price
                trade_pnl = (entry_price - exit_price) * shares

            # Update capital
            self.current_capital += trade_pnl

            # Record trade
            trade_info = {
                "date": date,
                "direction": "Long" if prediction == 1 else "Short",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": shares,
                "pnl": trade_pnl,
                "return": (trade_pnl / trade_amount) * 100 if trade_amount > 0 else 0,
            }

            self.trades.append(trade_info)
            print(f"Trade executed: {trade_info}")  # Debug print

        except Exception as e:
            print(f"Trade execution error: {str(e)}")

    def run_backtest(self, start_date, end_date, lookback_days=10):
        """Run backtest with debugging information"""
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            # Fetch data
            data_start = start_date - timedelta(days=lookback_days)
            full_data = self.data_handler.fetch_data(data_start, end_date)

            if full_data.empty:
                raise ValueError(f"No data available for {self.symbol}")

            print("\nRunning backtest...")
            self.model.feature_names = self.data_handler.get_feature_names()

            # Iterate through backtest period
            trading_days = full_data[start_date:end_date].index
            total_days = len(trading_days)

            print(f"Total trading days in backtest period: {total_days}")

            for i in range(lookback_days, total_days):
                current_date = trading_days[i]
                print(f"\nProcessing day {i+1}/{total_days}: {current_date}")

                # Get historical data up to current date
                historical_data = full_data.iloc[i - lookback_days : i]

                # Calculate features
                features = self.data_handler.calculate_features(historical_data)

                if len(features) >= lookback_days:
                    # Prepare training data
                    X_train, y_train = self.model.prepare_data(
                        features[:-1], historical_data[:-1]
                    )

                    # Prepare current day's features
                    X_current = features.iloc[[-1]][self.model.feature_names]

                    # Get prediction
                    prediction = self.model.train_and_predict(
                        X_train, y_train, X_current
                    )

                    # Execute trade
                    current_prices = full_data.loc[current_date]
                    self._execute_trade(current_date, prediction, current_prices)
                else:
                    print(f"Insufficient data for training: {len(features)} days")

                # Record daily portfolio value
                self.daily_values.append(
                    {"date": current_date, "portfolio_value": self.current_capital}
                )

            print(f"\nBacktest completed. Processed {len(trading_days)} trading days.")
            print(f"Final capital: ${self.current_capital:,.2f}")
            print(f"Total trades executed: {len(self.trades)}")

        except Exception as e:
            print(f"Backtest error: {str(e)}")
            raise

    def get_statistics(self):
        """Calculate and return backtest statistics"""
        if not self.daily_values:
            return {
                "Initial Capital": self.initial_capital,
                "Final Capital": self.current_capital,
                "Total Return %": 0.0,
                "Total Trades": 0,
                "Winning Trades": 0,
                "Win Rate %": 0.0,
                "Average Trade Return %": 0.0,
                "Best Trade %": 0.0,
                "Worst Trade %": 0.0,
                "Sharpe Ratio": 0.0,
                "Max Drawdown %": 0.0,
            }

        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        portfolio_values = pd.DataFrame(self.daily_values)

        stats = {
            "Initial Capital": self.initial_capital,
            "Final Capital": self.current_capital,
            "Total Return %": ((self.current_capital / self.initial_capital) - 1) * 100,
            "Total Trades": len(self.trades),
            "Winning Trades": (
                len(trades_df[trades_df["pnl"] > 0]) if not trades_df.empty else 0
            ),
            "Win Rate %": (
                (len(trades_df[trades_df["pnl"] > 0]) / len(trades_df) * 100)
                if not trades_df.empty
                else 0
            ),
            "Average Trade Return %": (
                trades_df["return"].mean() if not trades_df.empty else 0
            ),
            "Best Trade %": trades_df["return"].max() if not trades_df.empty else 0,
            "Worst Trade %": trades_df["return"].min() if not trades_df.empty else 0,
            "Sharpe Ratio": self._calculate_sharpe_ratio(portfolio_values),
            "Max Drawdown %": self._calculate_max_drawdown(portfolio_values),
        }

        return stats

    def _calculate_sharpe_ratio(self, portfolio_values):
        """Calculate Sharpe Ratio"""
        daily_returns = portfolio_values["portfolio_value"].pct_change().dropna()
        if len(daily_returns) < 2:
            return 0
        return (
            np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            if daily_returns.std() != 0
            else 0
        )

    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate Maximum Drawdown"""
        values = portfolio_values["portfolio_value"].values
        peak = values[0]
        max_drawdown = 0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown * 100

    def plot_results(self):
        """Plot backtest results"""
        if not self.daily_values:
            print("No data to plot")
            return

        portfolio_values = pd.DataFrame(self.daily_values)
        trades_df = pd.DataFrame(self.trades)

        plt.style.use("default")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="#f0f0f0")
        fig.suptitle(f"{self.symbol} Trading Strategy Results", fontsize=16)

        # Portfolio Value
        axes[0, 0].plot(
            portfolio_values["date"],
            portfolio_values["portfolio_value"],
            color="blue",
            linewidth=2,
        )
        axes[0, 0].set_title("Portfolio Value Over Time")
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Portfolio Value ($)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Trade Returns Distribution
        if not trades_df.empty:
            axes[0, 1].hist(
                trades_df["return"], bins=30, color="green", alpha=0.6, density=True
            )
            axes[0, 1].set_title("Trade Returns Distribution")
            axes[0, 1].set_xlabel("Return (%)")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].grid(True, alpha=0.3)

        # Cumulative Returns
        cumulative_returns = portfolio_values["portfolio_value"].pct_change().cumsum()
        axes[1, 0].plot(
            portfolio_values["date"], cumulative_returns, color="red", linewidth=2
        )
        axes[1, 0].set_title("Cumulative Returns")
        axes[1, 0].set_xlabel("Date")
        axes[1, 0].set_ylabel("Cumulative Return")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Trade Outcomes
        if not trades_df.empty:
            outcomes = trades_df["pnl"].apply(lambda x: "Win" if x > 0 else "Loss")
            win_count = (outcomes == "Win").sum()
            loss_count = (outcomes == "Loss").sum()

            axes[1, 1].bar(
                ["Wins", "Losses"],
                [win_count, loss_count],
                color=["green", "red"],
                alpha=0.6,
            )
            axes[1, 1].set_title("Trade Outcomes")
            axes[1, 1].set_xlabel("Outcome")
            axes[1, 1].set_ylabel("Number of Trades")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def save_results(self, filename):
        """Save backtest results to CSV"""
        if not self.trades:
            print("No trades to save")
            return

        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(f"{filename}_trades.csv", index=False)

        portfolio_df = pd.DataFrame(self.daily_values)
        portfolio_df.to_csv(f"{filename}_portfolio.csv", index=False)

        stats = self.get_statistics()
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(f"{filename}_statistics.csv", index=False)


# Example usage
if __name__ == "__main__":
    trader = DayTrader(
        symbol="MSFT", initial_capital=100000, position_size=0.95, stop_loss=0.02
    )

    try:
        # Run backtest
        trader.run_backtest(
            start_date="2024-01-01",
            end_date="2024-07-14",
            lookback_days=20,
        )

        # Print statistics
        print("\nBacktest Statistics:")
        stats = trader.get_statistics()
        for key, value in stats.items():
            print(
                f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
            )

        # Plot results
        trader.plot_results()

        # Save results
        trader.save_results("MSFT_backtest_results")

    except Exception as e:
        print(f"Error running backtest: {str(e)}")
