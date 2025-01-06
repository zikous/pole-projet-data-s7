from dataclasses import dataclass
from typing import Dict


@dataclass
class PortfolioPerformance:
    """Stores portfolio performance metrics"""

    return_value: float
    risk: float
    sharpe_ratio: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "return": self.return_value,
            "risk": self.risk,
            "sharpe_ratio": self.sharpe_ratio,
        }
