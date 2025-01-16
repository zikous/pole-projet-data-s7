import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class MarketDataHandler:
    # MarketDataHandler class remains unchanged
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = None

    def fetch_data(self, start_date, end_date, additional_lookback_days=40):
        try:
            adjusted_start = pd.to_datetime(start_date) - timedelta(
                days=additional_lookback_days
            )
            self.data = yf.download(self.symbol, start=adjusted_start, end=end_date)

            if self.data.empty:
                raise ValueError(f"No data available for {self.symbol}")

            print(f"Fetched {len(self.data)} days of data")
            return self.data

        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")

    def calculate_features(self, data_slice):
        features = pd.DataFrame(index=data_slice.index)

        features["Open-Close"] = (
            data_slice["Open"] - data_slice["Close"]
        ) / data_slice["Open"]
        features["High-Low"] = (data_slice["High"] - data_slice["Low"]) / data_slice[
            "Low"
        ]
        features["price_momentum"] = data_slice["Close"].pct_change()
        features["volume_momentum"] = data_slice["Volume"].pct_change()

        ma5 = data_slice["Close"].rolling(window=5).mean()
        ma10 = data_slice["Close"].rolling(window=10).mean()

        features["MA_ratio"] = ma5 / ma10
        features["volatility"] = data_slice["Close"].rolling(5).std()
        features["price_ma5_ratio"] = data_slice["Close"] / ma5

        return features.ffill().fillna(0)

    def get_feature_names(self):
        return [
            "Open-Close",
            "High-Low",
            "price_momentum",
            "volume_momentum",
            "MA_ratio",
            "volatility",
            "price_ma5_ratio",
        ]


class ModelFactory:
    """New class to create fresh model instances"""

    @staticmethod
    def create_model(n_estimators=100, max_depth=3, min_samples_split=5):
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
        )


class TradingModel:
    def __init__(self, n_estimators=100, max_depth=3, min_samples_split=5):
        self.model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
        }
        self.feature_names = None

    def prepare_data(self, features, prices):
        X = features[self.feature_names]
        returns = prices["Close"].pct_change().shift(-1)
        y = np.where(returns > 0.001, 1, np.where(returns < -0.001, -1, 0))[:-1]
        return X[:-1], y.ravel()

    def train_and_predict(self, X_train, y_train, X_predict):
        if len(X_train) < 10:
            print(f"Insufficient training data: {len(X_train)} samples")
            return 0

        try:
            # Create a fresh model instance for each prediction
            model = ModelFactory.create_model(**self.model_params)
            model.fit(X_train, y_train)
            prediction = model.predict(X_predict)[0]
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
        if prediction == 0:
            return

        try:
            current_price = float(prices["Open"].iloc[0])
            close_price = float(prices["Close"].iloc[0])
            high_price = float(prices["High"].iloc[0])
            low_price = float(prices["Low"].iloc[0])

            trade_amount = self.current_capital * self.position_size
            shares = int(trade_amount / current_price)

            if shares == 0:
                return

            entry_price = current_price
            exit_price = close_price
            stop_loss_price = entry_price * (
                1 - self.stop_loss if prediction == 1 else 1 + self.stop_loss
            )

            if prediction == 1:
                if low_price <= stop_loss_price:
                    exit_price = stop_loss_price
                trade_pnl = (exit_price - entry_price) * shares
            else:
                if high_price >= stop_loss_price:
                    exit_price = stop_loss_price
                trade_pnl = (entry_price - exit_price) * shares

            self.current_capital += trade_pnl

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
            print(
                f"Trade executed on {date}: {trade_info['direction']} | PnL: ${trade_pnl:.2f}"
            )

        except Exception as e:
            print(f"Trade execution error: {str(e)}")

    def run_backtest(self, start_date, end_date, lookback_days=20):
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            full_data = self.data_handler.fetch_data(start_date, end_date)

            if full_data.empty:
                raise ValueError(f"No data available for {self.symbol}")

            print("\nRunning backtest...")
            self.model.feature_names = self.data_handler.get_feature_names()

            backtest_data = full_data[start_date:end_date]
            all_dates = full_data.index
            total_days = len(backtest_data)

            print(f"Total trading days in backtest period: {total_days}")

            for i in range(total_days):
                current_date = backtest_data.index[i]
                current_idx = all_dates.get_loc(current_date)

                start_idx = max(0, current_idx - lookback_days)
                historical_data = full_data.iloc[start_idx : current_idx + 1]

                features = self.data_handler.calculate_features(historical_data)

                if len(features) >= lookback_days:
                    X_train, y_train = self.model.prepare_data(
                        features[:-1], historical_data[:-1]
                    )
                    X_current = features.iloc[[-1]][self.model.feature_names]
                    prediction = self.model.train_and_predict(
                        X_train, y_train, X_current
                    )
                    current_prices = backtest_data.iloc[[i]]
                    self._execute_trade(current_date, prediction, current_prices)

                self.daily_values.append(
                    {"date": current_date, "portfolio_value": self.current_capital}
                )

            print(f"\nBacktest completed. Final capital: ${self.current_capital:,.2f}")

        except Exception as e:
            print(f"Backtest error: {str(e)}")
            raise

    # Rest of the DayTrader class methods remain unchanged
    def get_statistics(self):
        if not self.daily_values:
            return {
                "Initial Capital": self.initial_capital,
                "Final Capital": self.current_capital,
                "Total Return %": 0.0,
                "Total Trades": 0,
                "Win Rate %": 0.0,
                "Average Trade Return %": 0.0,
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
            "Win Rate %": (
                (len(trades_df[trades_df["pnl"] > 0]) / len(trades_df) * 100)
                if not trades_df.empty
                else 0
            ),
            "Average Trade Return %": (
                trades_df["return"].mean() if not trades_df.empty else 0
            ),
            "Sharpe Ratio": self._calculate_sharpe_ratio(portfolio_values),
            "Max Drawdown %": self._calculate_max_drawdown(portfolio_values),
        }

        return stats

    def _calculate_sharpe_ratio(self, portfolio_values):
        daily_returns = portfolio_values["portfolio_value"].pct_change().dropna()
        if len(daily_returns) < 2:
            return 0
        return (
            np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            if daily_returns.std() != 0
            else 0
        )

    def _calculate_max_drawdown(self, portfolio_values):
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
        if not self.daily_values:
            print("No data to plot")
            return

        portfolio_values = pd.DataFrame(self.daily_values)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f"{self.symbol} Trading Strategy Performance", y=0.95)

        ax1.plot(
            portfolio_values["date"],
            portfolio_values["portfolio_value"],
            color="blue",
            linewidth=2,
        )
        ax1.set_title("Portfolio Value Over Time")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        if not trades_df.empty:
            ax2.hist(trades_df["return"], bins=30, color="green", alpha=0.6)
            ax2.axvline(x=0, color="red", linestyle="--", alpha=0.5)
            ax2.set_title("Trade Returns Distribution")
            ax2.set_xlabel("Return (%)")
            ax2.set_ylabel("Frequency")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_results(self, filename):
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


if __name__ == "__main__":
    trader = DayTrader(
        symbol="AAPL", initial_capital=1000000, position_size=0.95, stop_loss=0.02
    )

    try:
        trader.run_backtest(
            start_date="2018-01-01", end_date="2020-03-14", lookback_days=20
        )

        print("\nBacktest Statistics:")
        stats = trader.get_statistics()
        for key, value in stats.items():
            print(
                f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
            )

        trader.plot_results()
        trader.save_results("MSFT_backtest_results")

    except Exception as e:
        print(f"Error running backtest: {str(e)}")
