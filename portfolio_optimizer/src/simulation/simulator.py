from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from ..config.settings import PortfolioConfig, ANNUALIZATION_FACTOR
from ..data.data_loader import DataLoader
from ..optimization.portfolio_optimizer import PortfolioOptimizer
import numpy as np
from typing import List, Dict, Tuple, Optional
from ..optimization.performance import PortfolioPerformance
import pandas as pd


class PortfolioSimulator:
    """Simulates and compares different portfolio strategies"""

    def __init__(self, config: PortfolioConfig):
        """Initialize simulator with configuration"""
        self.config = config
        self.data_loader = DataLoader()
        self.optimizer = PortfolioOptimizer()
        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize data storage containers"""
        self.dates: List[datetime] = []
        self.portfolio_values: Dict[str, List[float]] = {
            strategy: []
            for strategy in ["sharpe", "equal_weight", "min_risk", "target_return"]
        }
        self.weights: Dict[str, List[np.ndarray]] = {
            strategy: []
            for strategy in ["sharpe", "equal_weight", "min_risk", "target_return"]
        }
        self.closing_prices_history: List[np.ndarray] = []
        self.performance_history: Dict[str, List[PortfolioPerformance]] = {
            strategy: []
            for strategy in ["sharpe", "equal_weight", "min_risk", "target_return"]
        }

    def get_date_range(self, current_month: int) -> Tuple[datetime, datetime]:
        """Calculate start and end dates for the current iteration"""
        end_date = datetime(self.config.start_year, 1, 1) + relativedelta(
            months=current_month
        )
        start_date = (
            end_date.replace(day=1) - relativedelta(months=self.config.lookback_months)
        ).replace(day=1)
        return start_date, end_date

    def calculate_portfolio_weights(
        self, daily_returns: pd.DataFrame, risk_free_rate: float
    ) -> Dict[str, np.ndarray]:
        """Calculate weights for all portfolio strategies"""
        num_assets = len(self.config.tickers)

        weights = {
            "sharpe": self.optimizer.optimize_max_sharpe_ratio(
                daily_returns, num_assets, risk_free_rate
            ).x,
            "equal_weight": np.ones(num_assets) / num_assets,
            "min_risk": self.optimizer.optimize_min_risk(daily_returns, num_assets).x,
        }

        if self.config.target_return is not None:
            target_result = self.optimizer.optimize_for_target_return(
                daily_returns, num_assets, self.config.target_return
            )
            weights["target_return"] = target_result.x

        return weights

    def _calculate_portfolio_value(
        self, month: int, strategy: str, weights: np.ndarray, current_prices: np.ndarray
    ) -> float:
        """Calculate portfolio value for a given strategy"""
        if month == 0:
            shares = self.config.initial_capital * weights / current_prices
            return np.sum(shares * current_prices)
        else:
            prev_value = self.portfolio_values[strategy][month - 1]
            prev_weights = self.weights[strategy][month - 1]
            prev_prices = self.closing_prices_history[month - 1]
            shares = prev_value * prev_weights / prev_prices
            return np.sum(shares * current_prices)

    def update_portfolio_values(
        self,
        month: int,
        weights: Dict[str, np.ndarray],
        current_prices: np.ndarray,
        daily_returns: pd.DataFrame,
        risk_free_rate: float,
    ):
        """Update portfolio values and performance metrics for all strategies"""
        for strategy in self.portfolio_values.keys():
            if strategy == "target_return" and self.config.target_return is None:
                continue

            portfolio_value = self._calculate_portfolio_value(
                month, strategy, weights[strategy], current_prices
            )
            self.portfolio_values[strategy].append(portfolio_value)
            self.weights[strategy].append(weights[strategy])

            performance = self.optimizer.calculate_portfolio_performance(
                weights[strategy], daily_returns, risk_free_rate
            )
            self.performance_history[strategy].append(performance)

    def run_simulation(self):
        """Run the portfolio simulation"""
        for month in range(self.config.total_months):
            start_date, end_date = self.get_date_range(month)
            closing_prices = self.data_loader.load_data(
                self.config.tickers,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )

            daily_returns = self.data_loader.calculate_daily_returns(closing_prices)
            risk_free_rate = self.data_loader.get_risk_free_rate(
                start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )

            weights = self.calculate_portfolio_weights(daily_returns, risk_free_rate)
            current_prices = closing_prices.iloc[-1].values

            self.dates.append(end_date)
            self.closing_prices_history.append(current_prices)
            self.update_portfolio_values(
                month, weights, current_prices, daily_returns, risk_free_rate
            )

    def plot_results(self):
        """Plot portfolio values over time for all strategies"""
        plt.figure(figsize=(12, 6))
        dates = pd.to_datetime(self.dates)
        styles = {
            "sharpe": "-",
            "equal_weight": "--",
            "min_risk": "-.",
            "target_return": ":",
        }

        for strategy, values in self.portfolio_values.items():
            if strategy == "target_return" and self.config.target_return is None:
                continue
            plt.plot(
                dates,
                values,
                label=strategy.replace("_", " ").title(),
                linestyle=styles[strategy],
            )

        plt.title("Portfolio Value Over Time Comparison")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value (USD)")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive summary statistics for each strategy"""
        summary = {}
        for strategy in self.portfolio_values.keys():
            if strategy == "target_return" and self.config.target_return is None:
                continue

            values = np.array(self.portfolio_values[strategy])
            monthly_returns = np.diff(values) / values[:-1]

            # Calculate drawdown metrics
            peak = np.maximum.accumulate(values)
            drawdown = (peak - values) / peak
            max_drawdown = np.max(drawdown)

            # Calculate return metrics
            total_years = self.config.total_months / 12
            total_return = (values[-1] / values[0]) - 1
            annualized_return = (1 + total_return) ** (1 / total_years) - 1

            # Calculate risk metrics
            monthly_volatility = np.std(monthly_returns)
            annualized_volatility = monthly_volatility * ANNUALIZATION_FACTOR

            summary[strategy] = {
                "final_value": values[-1],
                "total_return": total_return,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": (
                    annualized_return / annualized_volatility
                    if annualized_volatility > 0
                    else 0
                ),
                "calmar_ratio": (
                    annualized_return / max_drawdown if max_drawdown > 0 else 0
                ),
            }

        return summary
