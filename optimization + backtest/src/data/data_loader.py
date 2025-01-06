import yfinance as yf
import pandas as pd
from ..config.settings import DEFAULT_RISK_FREE_RATE
from typing import List


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
