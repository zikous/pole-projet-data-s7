import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize, OptimizeResult

# Constants
TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.02
ANNUALIZATION_FACTOR = np.sqrt(12)  # For monthly to annual conversion


@dataclass
class PortfolioConfig:
    """Configuration settings for portfolio optimization"""

    initial_capital: float
    lookback_months: int
    total_months: int
    start_year: int
    tickers: List[str]
    target_return: Optional[float] = None


@dataclass
class PortfolioPerformance:
    """Stores portfolio performance metrics"""

    return_value: float
    risk: float
    sharpe_ratio: float

    def to_dict(self) -> Dict[str, float]:
        """Convert performance metrics to dictionary"""
        return {
            "return": self.return_value,
            "risk": self.risk,
            "sharpe_ratio": self.sharpe_ratio,
        }


class DataLoader:
    """Handles data loading and processing operations"""

    @staticmethod
    def load_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load closing price data for given tickers

        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame of closing prices
        """
        try:
            data = yf.download(tickers, start=start_date, end=end_date)
            if data.empty:
                raise ValueError("No data retrieved for the specified period")
            return data["Close"]
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    @staticmethod
    def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns from price data"""
        return prices.pct_change().dropna()

    @staticmethod
    def get_risk_free_rate(start_date: str, end_date: str) -> float:
        """
        Fetch and calculate average risk-free rate from Treasury yield data
        Returns default rate if data unavailable
        """
        try:
            risk_free_data = yf.download("^TNX", start=start_date, end=end_date)
            if risk_free_data.empty:
                print(
                    f"Warning: Using default risk-free rate: {DEFAULT_RISK_FREE_RATE}"
                )
                return DEFAULT_RISK_FREE_RATE

            average_yield = risk_free_data["Close"].mean() / 100
            print(f"Risk-Free Rate ({start_date} to {end_date}): {average_yield:.4f}")
            return average_yield
        except Exception as e:
            print(f"Warning: Using default rate due to error: {e}")
            return DEFAULT_RISK_FREE_RATE


class PortfolioOptimizer:
    """Handles portfolio optimization strategies"""

    @staticmethod
    def generate_random_weights(num_assets: int) -> np.ndarray:
        """
        Generate random weights that sum to 1

        Args:
            num_assets: Number of assets in portfolio

        Returns:
            Array of random weights summing to 1
        """
        return np.random.dirichlet(np.ones(num_assets))

    @staticmethod
    def calculate_portfolio_performance(
        weights: np.ndarray, daily_returns: pd.DataFrame, risk_free_rate: float
    ) -> PortfolioPerformance:
        """
        Calculate portfolio performance metrics

        Args:
            weights: Array of portfolio weights
            daily_returns: DataFrame of daily returns
            risk_free_rate: Annual risk-free rate (as decimal)

        Returns:
            PortfolioPerformance object with return, risk, and Sharpe ratio
        """
        # Calculate annualized metrics
        annualized_returns = (1 + daily_returns.mean()) ** TRADING_DAYS_PER_YEAR - 1
        annualized_covariance = daily_returns.cov() * TRADING_DAYS_PER_YEAR

        portfolio_return = np.sum(annualized_returns * weights)
        portfolio_risk = np.sqrt(
            np.dot(weights.T, np.dot(annualized_covariance, weights))
        )

        # Calculate Sharpe ratio with safety check
        sharpe_ratio = (
            (portfolio_return - risk_free_rate) / portfolio_risk
            if portfolio_risk > 0
            else 0
        )

        return PortfolioPerformance(portfolio_return, portfolio_risk, sharpe_ratio)

    @staticmethod
    def optimize_max_sharpe_ratio(
        daily_returns: pd.DataFrame,
        num_assets: int,
        risk_free_rate: float,
        initial_weights: Optional[np.ndarray] = None,
    ) -> OptimizeResult:
        """
        Optimize portfolio weights for maximum Sharpe ratio

        Args:
            daily_returns: DataFrame of daily returns
            num_assets: Number of assets in portfolio
            risk_free_rate: Annual risk-free rate (as decimal)
            initial_weights: Optional starting weights for optimization

        Returns:
            OptimizeResult from optimization
        """

        def negative_sharpe_ratio(weights: np.ndarray) -> float:
            perf = PortfolioOptimizer.calculate_portfolio_performance(
                weights, daily_returns, risk_free_rate
            )
            return -perf.sharpe_ratio

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(num_assets)]

        start_weights = (
            initial_weights
            if initial_weights is not None
            else PortfolioOptimizer.generate_random_weights(num_assets)
        )

        try:
            result = minimize(
                negative_sharpe_ratio,
                start_weights,
                method="SLSQP",
                constraints=constraints,
                bounds=bounds,
                options={"ftol": 1e-8, "maxiter": 1000},
            )

            if not result.success:
                print(f"Warning: Sharpe ratio optimization failed: {result.message}")

            return result

        except Exception as e:
            print(f"Error in Sharpe ratio optimization: {str(e)}")
            weights = np.ones(num_assets) / num_assets
            return OptimizeResult(
                x=weights,
                success=False,
                message=str(e),
                fun=negative_sharpe_ratio(weights),
            )

    @staticmethod
    def optimize_min_risk(
        daily_returns: pd.DataFrame,
        num_assets: int,
        initial_weights: Optional[np.ndarray] = None,
    ) -> OptimizeResult:
        """
        Optimize portfolio weights for minimum risk

        Args:
            daily_returns: DataFrame of daily returns
            num_assets: Number of assets in portfolio
            initial_weights: Optional starting weights for optimization

        Returns:
            OptimizeResult from optimization
        """

        def portfolio_risk(weights: np.ndarray) -> float:
            return PortfolioOptimizer.calculate_portfolio_performance(
                weights, daily_returns, 0
            ).risk

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(num_assets)]

        start_weights = (
            initial_weights
            if initial_weights is not None
            else PortfolioOptimizer.generate_random_weights(num_assets)
        )

        try:
            result = minimize(
                portfolio_risk,
                start_weights,
                method="SLSQP",
                constraints=constraints,
                bounds=bounds,
                options={"ftol": 1e-8, "maxiter": 1000},
            )

            if not result.success:
                print(f"Warning: Minimum risk optimization failed: {result.message}")

            return result

        except Exception as e:
            print(f"Error in minimum risk optimization: {str(e)}")
            weights = np.ones(num_assets) / num_assets
            return OptimizeResult(
                x=weights, success=False, message=str(e), fun=portfolio_risk(weights)
            )

    @staticmethod
    def optimize_for_target_return(
        daily_returns: pd.DataFrame,
        num_assets: int,
        target_return: float,
        initial_weights: Optional[np.ndarray] = None,
    ) -> OptimizeResult:
        """
        Optimize portfolio weights to minimize risk for a given target return

        Args:
            daily_returns: DataFrame of daily returns
            num_assets: Number of assets in portfolio
            target_return: Target annualized return (as decimal)
            initial_weights: Optional starting weights for optimization

        Returns:
            OptimizeResult from optimization
        """

        def portfolio_risk(weights: np.ndarray) -> float:
            return PortfolioOptimizer.calculate_portfolio_performance(
                weights, daily_returns, 0
            ).risk

        def return_constraint(weights: np.ndarray) -> float:
            portfolio_return = PortfolioOptimizer.calculate_portfolio_performance(
                weights, daily_returns, 0
            ).return_value
            return portfolio_return - target_return

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Weights sum to 1
            {"type": "ineq", "fun": return_constraint},  # Return >= target
        ]
        bounds = [(0.0, 1.0) for _ in range(num_assets)]

        start_weights = (
            initial_weights
            if initial_weights is not None
            else PortfolioOptimizer.generate_random_weights(num_assets)
        )

        try:
            # Try multiple random starting points if optimization fails
            for _ in range(5):
                result = minimize(
                    portfolio_risk,
                    start_weights,
                    method="SLSQP",
                    constraints=constraints,
                    bounds=bounds,
                    options={"ftol": 1e-8, "maxiter": 1000},
                )

                if result.success:
                    return result

            # If all attempts fail, try with relaxed constraints
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {
                    "type": "ineq",
                    "fun": lambda w: return_constraint(w) + 0.01,
                },  # Add 1% tolerance
            ]

            result = minimize(
                portfolio_risk,
                start_weights,
                method="SLSQP",
                constraints=constraints,
                bounds=bounds,
                options={"ftol": 1e-8, "maxiter": 1000},
            )

            if not result.success:
                print(f"Warning: Target return optimization failed: {result.message}")

            return result

        except Exception as e:
            print(f"Error in target return optimization: {str(e)}")
            weights = np.ones(num_assets) / num_assets
            return OptimizeResult(
                x=weights,
                success=False,
                message=str(e),
                fun=portfolio_risk(weights),
            )


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
