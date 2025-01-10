from services.utils import (
    PortfolioConfig,
    PortfolioSimulator,
    DataLoader,
    PortfolioOptimizer,
)
import numpy as np
from services.chart_service import create_portfolio_chart
import logging


def run_optimization(
    tickers, initial_capital, start_date, end_date, target_return=None
):
    # Initialize a logger for debugging purposes
    logger = logging.getLogger(__name__)

    # Log the function parameters for debugging
    logger.debug(
        f"""
    Running optimization with:
    - tickers: {tickers}
    - initial_capital: {initial_capital}
    - start_date: {start_date}
    - end_date: {end_date}
    - target_return: {target_return}
    """
    )

    try:
        # Step 1: Load the stock price data
        data_loader = DataLoader()  # Instantiate a data loader object
        prices = data_loader.load_data(
            tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )  # Load the price data for the specified tickers and date range

        # Check if the price data is empty and raise an error if true
        if prices.empty:
            raise ValueError("No price data found for the given tickers and date range")

        # Step 2: Calculate daily returns based on price data
        daily_returns = data_loader.calculate_daily_returns(
            prices
        )  # Calculate daily returns
        risk_free_rate = data_loader.get_risk_free_rate(
            start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )  # Get the risk-free rate for the specified date range

        # Step 3: Instantiate a portfolio optimizer object
        optimizer = PortfolioOptimizer()
        num_assets = len(tickers)  # Get the number of assets (stocks)

        # Initialize dictionaries to store the weights and performance of different strategies
        weights = {}
        performance = {}

        # Step 4: Perform optimization for various strategies

        # Maximize Sharpe ratio (optimize for highest return per unit of risk)
        sharpe_result = optimizer.optimize_max_sharpe_ratio(
            daily_returns, num_assets, risk_free_rate
        )
        if not hasattr(sharpe_result, "x"):
            raise ValueError("Sharpe ratio optimization failed")
        weights["sharpe"] = (
            sharpe_result.x.tolist()
        )  # Store the weights for the Sharpe ratio strategy

        # Equal-weight portfolio (assign equal weights to each asset)
        equal_weights = np.ones(num_assets) / num_assets
        weights["equal_weight"] = equal_weights.tolist()

        # Minimize risk (optimize for the lowest volatility portfolio)
        min_risk_result = optimizer.optimize_min_risk(daily_returns, num_assets)
        if not hasattr(min_risk_result, "x"):
            raise ValueError("Minimum risk optimization failed")
        weights["min_risk"] = (
            min_risk_result.x.tolist()
        )  # Store the weights for the minimum risk strategy

        # Target return optimization (optimize for a specific target return)
        if target_return is not None:
            target_result = optimizer.optimize_for_target_return(
                daily_returns, num_assets, target_return
            )
            if not hasattr(target_result, "x"):
                raise ValueError("Target return optimization failed")
            weights["target_return"] = (
                target_result.x.tolist()
            )  # Store the weights for the target return strategy

        # Step 5: Calculate performance metrics for each strategy
        for strategy, w in weights.items():
            w_array = np.array(w)  # Convert the weights to a numpy array
            perf = optimizer.calculate_portfolio_performance(
                w_array, daily_returns, risk_free_rate
            )  # Calculate the portfolio performance (return, risk, Sharpe ratio)
            performance[strategy] = {
                "return": float(perf.return_value),
                "risk": float(perf.risk),
                "sharpe_ratio": float(perf.sharpe_ratio),
            }

        # Step 6: Compile the optimization results into a dictionary
        result = {"weights": weights, "performance": performance, "tickers": tickers}

        # Log the optimization results for debugging
        logger.debug(f"Optimization result: {result}")

        # Return the results
        return result

    except Exception as e:
        # Log the error if an exception occurs during the optimization process
        logger.error(f"Error in optimization calculation: {str(e)}", exc_info=True)

        # Re-raise the exception so that it can be handled elsewhere
        raise


def run_backtest(data):
    # Step 1: Create a PortfolioConfig object with parameters from the input data
    config = PortfolioConfig(
        initial_capital=float(
            data["initial_capital"]
        ),  # Initial capital for the portfolio
        lookback_months=int(
            data["lookback_months"]
        ),  # Lookback period in months for historical data
        total_months=int(
            data["total_months"]
        ),  # Total period for backtesting in months
        start_year=int(data["start_year"]),  # Starting year for backtest
        tickers=data["tickers"],  # List of stock tickers to include in the portfolio
        target_return=(
            float(data["target_return"]) if data["target_return"] else None
        ),  # Target return if provided
    )

    # Step 2: Initialize the PortfolioSimulator with the created config
    simulator = PortfolioSimulator(config)

    # Step 3: Run the simulation using the simulator
    simulator.run_simulation()

    # Step 4: Generate a portfolio performance chart using the simulator's data
    chart = create_portfolio_chart(simulator.portfolio_values, simulator.dates)

    # Step 5: Return summary statistics and the generated chart
    return simulator.get_summary_statistics(), chart
