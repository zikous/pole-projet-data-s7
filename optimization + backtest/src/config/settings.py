from typing import List, Optional
from dataclasses import dataclass
import numpy as np

# Constants
TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.02
ANNUALIZATION_FACTOR = np.sqrt(12)


@dataclass
class PortfolioConfig:
    """Configuration settings for portfolio optimization"""

    initial_capital: float
    lookback_months: int
    total_months: int
    start_year: int
    tickers: List[str]
    target_return: Optional[float] = None
