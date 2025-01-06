import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.config.settings import PortfolioConfig
from src.simulation.simulator import PortfolioSimulator


def main():
    """Example usage of the portfolio optimization system"""
    np.random.seed(42)  # For reproducibility

    # config = PortfolioConfig(
    #     initial_capital=100000,
    #     lookback_months=6,
    #     total_months=12 * 3,
    #     start_year=2021,
    #     tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD"],
    #     target_return=0.15,
    # )
    config = PortfolioConfig(
        initial_capital=100000,  # Starting with $100,000
        lookback_months=1,  # Only 1 month of data (extremely short, practically useless for strategy building)
        total_months=12 * 2,  # Only 1 year of total data (again, very short)
        start_year=2023,  # Very recent data (no history to smooth out volatility)
        tickers=[  # Picking volatile, struggling, and highly speculative assets
            "AMC",  # AMC Theatres (known for struggling financially, meme stock volatility)
            "GME",  # GameStop (memetic volatility, historically unreliable as an investment)
            "TLRY",  # Tilray (a marijuana stock that's been inconsistent and highly volatile)
        ],
        target_return=0.15,
    )

    try:
        simulator = PortfolioSimulator(config)
        simulator.run_simulation()
        simulator.plot_results()

        # Display results
        summary = simulator.get_summary_statistics()
        print("\nPortfolio Summary Statistics:")
        for strategy, stats in summary.items():
            print(f"\n{strategy.replace('_', ' ').title()} Strategy:")
            for metric, value in stats.items():
                if metric in ["final_value"]:
                    print(f"{metric.replace('_', ' ').title()}: ${value:,.2f}")
                else:
                    print(f"{metric.replace('_', ' ').title()}: {value*100:.2f}%")

    except Exception as e:
        print(f"Error in portfolio simulation: {str(e)}")


if __name__ == "__main__":
    main()
