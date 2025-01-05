# services/portfolio_service.py
from ok import PortfolioConfig, PortfolioSimulator, DataLoader, PortfolioOptimizer
import numpy as np
from services.chart_service import create_portfolio_chart
import logging


def run_optimization(
    tickers, initial_capital, start_date, end_date, target_return=None
):
    logger = logging.getLogger(__name__)
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
        data_loader = DataLoader()
        prices = data_loader.load_data(
            tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )

        if prices.empty:
            raise ValueError("No price data found for the given tickers and date range")

        daily_returns = data_loader.calculate_daily_returns(prices)
        risk_free_rate = data_loader.get_risk_free_rate(
            start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )

        optimizer = PortfolioOptimizer()
        num_assets = len(tickers)

        weights = {}
        performance = {}

        # Calculate weights for different strategies
        sharpe_result = optimizer.optimize_max_sharpe_ratio(
            daily_returns, num_assets, risk_free_rate
        )
        if not hasattr(sharpe_result, "x"):
            raise ValueError("Sharpe ratio optimization failed")
        weights["sharpe"] = sharpe_result.x.tolist()

        equal_weights = np.ones(num_assets) / num_assets
        weights["equal_weight"] = equal_weights.tolist()

        min_risk_result = optimizer.optimize_min_risk(daily_returns, num_assets)
        if not hasattr(min_risk_result, "x"):
            raise ValueError("Minimum risk optimization failed")
        weights["min_risk"] = min_risk_result.x.tolist()

        if target_return is not None:
            target_result = optimizer.optimize_for_target_return(
                daily_returns, num_assets, target_return
            )
            if not hasattr(target_result, "x"):
                raise ValueError("Target return optimization failed")
            weights["target_return"] = target_result.x.tolist()

        # Calculate performance metrics
        for strategy, w in weights.items():
            w_array = np.array(w)
            perf = optimizer.calculate_portfolio_performance(
                w_array, daily_returns, risk_free_rate
            )
            performance[strategy] = {
                "return": float(perf.return_value),
                "risk": float(perf.risk),
                "sharpe_ratio": float(perf.sharpe_ratio),
            }

        result = {"weights": weights, "performance": performance, "tickers": tickers}

        logger.debug(f"Optimization result: {result}")
        return result

    except Exception as e:
        logger.error(f"Error in optimization calculation: {str(e)}", exc_info=True)
        raise


def run_backtest(data):
    config = PortfolioConfig(
        initial_capital=float(data["initial_capital"]),
        lookback_months=int(data["lookback_months"]),
        total_months=int(data["total_months"]),
        start_year=int(data["start_year"]),
        tickers=data["tickers"],
        target_return=float(data["target_return"]) if data["target_return"] else None,
    )

    simulator = PortfolioSimulator(config)
    simulator.run_simulation()

    chart = create_portfolio_chart(simulator.portfolio_values, simulator.dates)

    return simulator.get_summary_statistics(), chart
