# Importation des modules nécessaires
from datetime import datetime  # Pour manipuler les dates
from dateutil.relativedelta import relativedelta  # Pour les calculs de dates
import matplotlib.pyplot as plt  # Pour les visualisations
from ..config.settings import (
    PortfolioConfig,
    ANNUALIZATION_FACTOR,
)  # Configuration et constantes
from ..data.data_loader import DataLoader  # Pour charger les données boursières
from ..optimization.portfolio_optimizer import (
    PortfolioOptimizer,
)  # Pour l'optimisation du portefeuille
import numpy as np  # Pour les calculs numériques
from typing import List, Dict, Tuple, Optional  # Pour le typage des données
from ..optimization.performance import (
    PortfolioPerformance,
)  # Pour stocker les métriques de performance
import pandas as pd  # Pour manipuler les données sous forme de DataFrame


class PortfolioSimulator:
    """
    Simule et compare différentes stratégies de portefeuille.
    Cette classe permet de simuler l'évolution d'un portefeuille en utilisant plusieurs stratégies d'optimisation :
    - Maximisation du ratio de Sharpe.
    - Poids égaux (égal répartition).
    - Minimisation du risque.
    - Atteinte d'un rendement cible.
    """

    def __init__(self, config: PortfolioConfig):
        """
        Initialise le simulateur avec une configuration donnée.

        Args:
            config (PortfolioConfig) : Configuration du portefeuille (capital initial, tickers, etc.).
        """
        self.config = config  # Configuration du portefeuille
        self.data_loader = DataLoader()  # Chargeur de données
        self.optimizer = PortfolioOptimizer()  # Optimiseur de portefeuille
        self._initialize_storage()  # Initialisation des conteneurs de données

    def _initialize_storage(self):
        """Initialise les conteneurs pour stocker les données de simulation."""
        self.dates: List[datetime] = []  # Dates de simulation
        self.portfolio_values: Dict[str, List[float]] = {
            strategy: []
            for strategy in ["sharpe", "equal_weight", "min_risk", "target_return"]
        }  # Valeurs du portefeuille pour chaque stratégie
        self.weights: Dict[str, List[np.ndarray]] = {
            strategy: []
            for strategy in ["sharpe", "equal_weight", "min_risk", "target_return"]
        }  # Poids des actifs pour chaque stratégie
        self.closing_prices_history: List[np.ndarray] = (
            []
        )  # Historique des prix de clôture
        self.performance_history: Dict[str, List[PortfolioPerformance]] = {
            strategy: []
            for strategy in ["sharpe", "equal_weight", "min_risk", "target_return"]
        }  # Historique des performances pour chaque stratégie

    def get_date_range(self, current_month: int) -> Tuple[datetime, datetime]:
        """
        Calcule les dates de début et de fin pour l'itération actuelle.

        Args:
            current_month (int) : Mois actuel de la simulation.

        Returns:
            Tuple[datetime, datetime] : Dates de début et de fin.
        """
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
        """
        Calcule les poids des actifs pour toutes les stratégies de portefeuille.

        Args:
            daily_returns (pd.DataFrame) : Rendements quotidiens des actifs.
            risk_free_rate (float) : Taux sans risque.

        Returns:
            Dict[str, np.ndarray] : Poids des actifs pour chaque stratégie.
        """
        num_assets = len(self.config.tickers)

        weights = {
            "sharpe": self.optimizer.optimize_max_sharpe_ratio(
                daily_returns, num_assets, risk_free_rate
            ).x,  # Maximisation du ratio de Sharpe
            "equal_weight": np.ones(num_assets) / num_assets,  # Poids égaux
            "min_risk": self.optimizer.optimize_min_risk(
                daily_returns, num_assets
            ).x,  # Minimisation du risque
        }

        if self.config.target_return is not None:
            # Optimisation pour atteindre un rendement cible
            target_result = self.optimizer.optimize_for_target_return(
                daily_returns, num_assets, self.config.target_return
            )
            weights["target_return"] = target_result.x

        return weights

    def _calculate_portfolio_value(
        self, month: int, strategy: str, weights: np.ndarray, current_prices: np.ndarray
    ) -> float:
        """
        Calcule la valeur du portefeuille pour une stratégie donnée.

        Args:
            month (int) : Mois actuel de la simulation.
            strategy (str) : Stratégie de portefeuille.
            weights (np.ndarray) : Poids des actifs.
            current_prices (np.ndarray) : Prix actuels des actifs.

        Returns:
            float : Valeur du portefeuille.
        """
        if month == 0:
            # Calcul initial des parts et de la valeur du portefeuille
            shares = self.config.initial_capital * weights / current_prices
            return np.sum(shares * current_prices)
        else:
            # Mise à jour de la valeur du portefeuille en fonction des prix actuels
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
        """
        Met à jour les valeurs du portefeuille et les métriques de performance pour toutes les stratégies.

        Args:
            month (int) : Mois actuel de la simulation.
            weights (Dict[str, np.ndarray]) : Poids des actifs pour chaque stratégie.
            current_prices (np.ndarray) : Prix actuels des actifs.
            daily_returns (pd.DataFrame) : Rendements quotidiens des actifs.
            risk_free_rate (float) : Taux sans risque.
        """
        for strategy in self.portfolio_values.keys():
            if strategy == "target_return" and self.config.target_return is None:
                continue

            # Calcul de la valeur du portefeuille
            portfolio_value = self._calculate_portfolio_value(
                month, strategy, weights[strategy], current_prices
            )
            self.portfolio_values[strategy].append(portfolio_value)
            self.weights[strategy].append(weights[strategy])

            # Calcul des métriques de performance
            performance = self.optimizer.calculate_portfolio_performance(
                weights[strategy], daily_returns, risk_free_rate
            )
            self.performance_history[strategy].append(performance)

    def run_simulation(self):
        """Exécute la simulation du portefeuille."""
        for month in range(self.config.total_months):
            # Calcul des dates de début et de fin pour l'itération actuelle
            start_date, end_date = self.get_date_range(month)

            # Chargement des prix de clôture et calcul des rendements quotidiens
            closing_prices = self.data_loader.load_data(
                self.config.tickers,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
            daily_returns = self.data_loader.calculate_daily_returns(closing_prices)

            # Récupération du taux sans risque
            risk_free_rate = self.data_loader.get_risk_free_rate(
                start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )

            # Calcul des poids des actifs pour chaque stratégie
            weights = self.calculate_portfolio_weights(daily_returns, risk_free_rate)
            current_prices = closing_prices.iloc[-1].values

            # Mise à jour des données de simulation
            self.dates.append(end_date)
            self.closing_prices_history.append(current_prices)
            self.update_portfolio_values(
                month, weights, current_prices, daily_returns, risk_free_rate
            )

    def plot_results(self):
        """Trace les valeurs du portefeuille au fil du temps pour toutes les stratégies."""
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

        plt.title("Évolution de la Valeur du Portefeuille")
        plt.xlabel("Date")
        plt.ylabel("Valeur du Portefeuille (USD)")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calcule des statistiques récapitulatives pour chaque stratégie.

        Returns:
            Dict[str, Dict[str, float]] : Statistiques récapitulatives pour chaque stratégie.
        """
        summary = {}
        for strategy in self.portfolio_values.keys():
            if strategy == "target_return" and self.config.target_return is None:
                continue

            values = np.array(self.portfolio_values[strategy])
            monthly_returns = np.diff(values) / values[:-1]

            # Calcul du drawdown (perte maximale)
            peak = np.maximum.accumulate(values)
            drawdown = (peak - values) / peak
            max_drawdown = np.max(drawdown)

            # Calcul des métriques de rendement
            total_years = self.config.total_months / 12
            total_return = (values[-1] / values[0]) - 1
            annualized_return = (1 + total_return) ** (1 / total_years) - 1

            # Calcul des métriques de risque
            monthly_volatility = np.std(monthly_returns)
            annualized_volatility = monthly_volatility * ANNUALIZATION_FACTOR

            # Calcul du ratio de Sharpe et du ratio de Calmar
            sharpe_ratio = (
                annualized_return / annualized_volatility
                if annualized_volatility > 0
                else 0
            )
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

            summary[strategy] = {
                "final_value": values[-1],  # Valeur finale du portefeuille
                "total_return": total_return,  # Rendement total
                "annualized_return": annualized_return,  # Rendement annualisé
                "annualized_volatility": annualized_volatility,  # Volatilité annualisée
                "max_drawdown": max_drawdown,  # Drawdown maximal
                "sharpe_ratio": sharpe_ratio,  # Ratio de Sharpe
                "calmar_ratio": calmar_ratio,  # Ratio de Calmar
            }

        return summary
