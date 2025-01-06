import numpy as np
from scipy.optimize import minimize, OptimizeResult
from ..config.settings import TRADING_DAYS_PER_YEAR
from .performance import PortfolioPerformance
import pandas as pd


class PortfolioOptimizer:
    """Handles portfolio optimization strategies"""

    @staticmethod
    def generate_random_weights(num_assets: int) -> np.ndarray:
        """Generate random weights that sum to 1"""
        return np.random.dirichlet(np.ones(num_assets))

    @staticmethod
    def calculate_portfolio_performance(
        weights: np.ndarray, daily_returns: pd.DataFrame, risk_free_rate: float
    ) -> PortfolioPerformance:
        """Calculate portfolio performance metrics"""
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
        daily_returns: pd.DataFrame, num_assets: int, risk_free_rate: float
    ) -> OptimizeResult:
        """Optimize portfolio weights for maximum Sharpe ratio"""

        def negative_sharpe_ratio(weights: np.ndarray) -> float:
            perf = PortfolioOptimizer.calculate_portfolio_performance(
                weights, daily_returns, risk_free_rate
            )
            return -perf.sharpe_ratio

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(num_assets)]

        return minimize(
            negative_sharpe_ratio,
            PortfolioOptimizer.generate_random_weights(num_assets),
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
        )

    @staticmethod
    def optimize_min_risk(
        daily_returns: pd.DataFrame, num_assets: int
    ) -> OptimizeResult:
        """Optimize portfolio weights for minimum risk"""

        def portfolio_risk(weights: np.ndarray) -> float:
            return PortfolioOptimizer.calculate_portfolio_performance(
                weights, daily_returns, 0
            ).risk

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(num_assets)]

        return minimize(
            portfolio_risk,
            PortfolioOptimizer.generate_random_weights(num_assets),
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
        )

    @staticmethod
    def optimize_for_target_return(
        daily_returns: pd.DataFrame, num_assets: int, target_return: float
    ) -> OptimizeResult:
        """
        Optimize portfolio weights to minimize risk for a given target return

        Args:
            daily_returns: DataFrame of daily returns
            num_assets: Number of assets in portfolio
            target_return: Target annualized return (as decimal)

        Returns:
            OptimizeResult from scipy.optimize.minimize
        """

        def portfolio_risk(weights: np.ndarray) -> float:
            return PortfolioOptimizer.calculate_portfolio_performance(
                weights, daily_returns, 0
            ).risk

        def return_constraint(weights: np.ndarray) -> float:
            portfolio_return = PortfolioOptimizer.calculate_portfolio_performance(
                weights, daily_returns, 0
            ).return_value
            # Add small tolerance to make optimization more stable
            return portfolio_return - target_return

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Weights sum to 1
            {"type": "ineq", "fun": return_constraint},  # Return >= target
        ]
        bounds = [(0.0, 1.0) for _ in range(num_assets)]

        try:
            # Try multiple random starting points if optimization fails
            for _ in range(5):
                result = minimize(
                    portfolio_risk,
                    PortfolioOptimizer.generate_random_weights(num_assets),
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
                PortfolioOptimizer.generate_random_weights(num_assets),
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
