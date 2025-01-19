import yfinance as yf  # Importe yfinance pour récupérer les données boursières
import numpy as np  # Importe numpy pour les calculs numériques
import pandas as pd  # Importe pandas pour manipuler les données
from datetime import datetime  # Importe datetime pour gérer les dates
from dateutil.relativedelta import (
    relativedelta,
)  # Importe relativedelta pour les calculs de dates
import matplotlib.pyplot as plt  # Importe matplotlib pour les graphiques
from dataclasses import dataclass  # Importe dataclass pour créer des classes de données
from typing import List, Dict, Tuple, Optional  # Importe les types pour les annotations
from scipy.optimize import (
    minimize,
    OptimizeResult,
)  # Importe minimize pour l'optimisation

# Constantes
TRADING_DAYS_PER_YEAR = 252  # Nombre de jours de trading par an
DEFAULT_RISK_FREE_RATE = 0.02  # Taux sans risque par défaut
ANNUALIZATION_FACTOR = np.sqrt(12)  # Facteur d'annualisation pour les calculs mensuels


@dataclass
class PortfolioConfig:
    """
    Configuration pour l'optimisation du portefeuille.
    Attributes:
        initial_capital (float): Capital initial du portefeuille.
        lookback_months (int): Nombre de mois pour l'historique des données.
        total_months (int): Nombre total de mois pour la simulation.
        start_year (int): Année de début de la simulation.
        tickers (List[str]): Liste des symboles boursiers.
        target_return (Optional[float]): Rendement cible (optionnel).
    """

    initial_capital: float
    lookback_months: int
    total_months: int
    start_year: int
    tickers: List[str]
    target_return: Optional[float] = None


@dataclass
class PortfolioPerformance:
    """
    Stocke les métriques de performance du portefeuille.
    Attributes:
        return_value (float): Rendement du portefeuille.
        risk (float): Risque du portefeuille.
        sharpe_ratio (float): Ratio de Sharpe du portefeuille.
    """

    return_value: float
    risk: float
    sharpe_ratio: float

    def to_dict(self) -> Dict[str, float]:
        """
        Convertit les métriques de performance en dictionnaire.
        Returns:
            Dict[str, float]: Dictionnaire contenant le rendement, le risque et le ratio de Sharpe.
        """
        return {
            "return": self.return_value,
            "risk": self.risk,
            "sharpe_ratio": self.sharpe_ratio,
        }


class DataLoader:
    """
    Gère le chargement et le traitement des données.
    """

    @staticmethod
    def load_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Charge les données de prix de clôture pour les tickers donnés.
        Args:
            tickers (List[str]): Liste des symboles boursiers.
            start_date (str): Date de début au format YYYY-MM-DD.
            end_date (str): Date de fin au format YYYY-MM-DD.
        Returns:
            pd.DataFrame: DataFrame des prix de clôture.
        """
        try:
            data = yf.download(
                tickers, start=start_date, end=end_date
            )  # Télécharge les données
            if data.empty:
                raise ValueError("Aucune donnée récupérée pour la période spécifiée")
            return data["Close"]  # Retourne uniquement les prix de clôture
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement des données : {str(e)}")

    @staticmethod
    def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les rendements quotidiens à partir des données de prix.
        Args:
            prices (pd.DataFrame): DataFrame des prix de clôture.
        Returns:
            pd.DataFrame: DataFrame des rendements quotidiens.
        """
        return (
            prices.pct_change().dropna()
        )  # Calcule les rendements et supprime les NaN

    @staticmethod
    def get_risk_free_rate(start_date: str, end_date: str) -> float:
        """
        Récupère et calcule le taux sans risque moyen à partir des données du Trésor américain.
        Retourne un taux par défaut si les données ne sont pas disponibles.
        Args:
            start_date (str): Date de début au format YYYY-MM-DD.
            end_date (str): Date de fin au format YYYY-MM-DD.
        Returns:
            float: Taux sans risque.
        """
        try:
            risk_free_data = yf.download(
                "^TNX", start=start_date, end=end_date
            )  # Télécharge les données du Trésor
            if risk_free_data.empty:
                print(
                    f"Avertissement : Utilisation du taux sans risque par défaut : {DEFAULT_RISK_FREE_RATE}"
                )
                return DEFAULT_RISK_FREE_RATE

            average_yield = (
                risk_free_data["Close"].mean() / 100
            )  # Calcule le rendement moyen
            print(f"Taux sans risque ({start_date} à {end_date}) : {average_yield:.4f}")
            return average_yield
        except Exception as e:
            print(
                f"Avertissement : Utilisation du taux par défaut en raison d'une erreur : {e}"
            )
            return DEFAULT_RISK_FREE_RATE


class PortfolioOptimizer:
    """
    Gère les stratégies d'optimisation du portefeuille.
    """

    @staticmethod
    def generate_random_weights(num_assets: int) -> np.ndarray:
        """
        Génère des poids aléatoires qui somment à 1.
        Args:
            num_assets (int): Nombre d'actifs dans le portefeuille.
        Returns:
            np.ndarray: Poids aléatoires.
        """
        return np.random.dirichlet(
            np.ones(num_assets)
        )  # Utilise la distribution de Dirichlet

    @staticmethod
    def calculate_portfolio_performance(
        weights: np.ndarray, daily_returns: pd.DataFrame, risk_free_rate: float
    ) -> PortfolioPerformance:
        """
        Calcule les métriques de performance du portefeuille.
        Args:
            weights (np.ndarray): Poids des actifs.
            daily_returns (pd.DataFrame): Rendements quotidiens.
            risk_free_rate (float): Taux sans risque.
        Returns:
            PortfolioPerformance: Métriques de performance.
        """
        # Calcule les métriques annualisées
        annualized_returns = (1 + daily_returns.mean()) ** TRADING_DAYS_PER_YEAR - 1
        annualized_covariance = daily_returns.cov() * TRADING_DAYS_PER_YEAR

        portfolio_return = np.sum(
            annualized_returns * weights
        )  # Rendement du portefeuille
        portfolio_risk = np.sqrt(
            np.dot(weights.T, np.dot(annualized_covariance, weights))
        )  # Risque du portefeuille

        # Calcule le ratio de Sharpe avec vérification de sécurité
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
        """
        Optimise les poids du portefeuille pour maximiser le ratio de Sharpe.
        Args:
            daily_returns (pd.DataFrame): Rendements quotidiens.
            num_assets (int): Nombre d'actifs.
            risk_free_rate (float): Taux sans risque.
        Returns:
            OptimizeResult: Résultat de l'optimisation.
        """

        def negative_sharpe_ratio(weights: np.ndarray) -> float:
            perf = PortfolioOptimizer.calculate_portfolio_performance(
                weights, daily_returns, risk_free_rate
            )
            return -perf.sharpe_ratio  # Minimise l'opposé du ratio de Sharpe

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]  # Contrainte : somme des poids = 1
        bounds = [(0.0, 1.0) for _ in range(num_assets)]  # Bornes : poids entre 0 et 1

        return minimize(
            negative_sharpe_ratio,
            PortfolioOptimizer.generate_random_weights(num_assets),
            method="SLSQP",  # Méthode d'optimisation
            constraints=constraints,
            bounds=bounds,
        )

    @staticmethod
    def optimize_min_risk(
        daily_returns: pd.DataFrame, num_assets: int
    ) -> OptimizeResult:
        """
        Optimise les poids du portefeuille pour minimiser le risque.
        Args:
            daily_returns (pd.DataFrame): Rendements quotidiens.
            num_assets (int): Nombre d'actifs.
        Returns:
            OptimizeResult: Résultat de l'optimisation.
        """

        def portfolio_risk(weights: np.ndarray) -> float:
            return PortfolioOptimizer.calculate_portfolio_performance(
                weights, daily_returns, 0
            ).risk  # Minimise le risque

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]  # Contrainte : somme des poids = 1
        bounds = [(0.0, 1.0) for _ in range(num_assets)]  # Bornes : poids entre 0 et 1

        return minimize(
            portfolio_risk,
            PortfolioOptimizer.generate_random_weights(num_assets),
            method="SLSQP",  # Méthode d'optimisation
            constraints=constraints,
            bounds=bounds,
        )

    @staticmethod
    def optimize_for_target_return(
        daily_returns: pd.DataFrame, num_assets: int, target_return: float
    ) -> OptimizeResult:
        """
        Optimise les poids du portefeuille pour minimiser le risque avec un rendement cible.
        Args:
            daily_returns (pd.DataFrame): Rendements quotidiens.
            num_assets (int): Nombre d'actifs.
            target_return (float): Rendement cible annualisé.
        Returns:
            OptimizeResult: Résultat de l'optimisation.
        """

        def portfolio_risk(weights: np.ndarray) -> float:
            return PortfolioOptimizer.calculate_portfolio_performance(
                weights, daily_returns, 0
            ).risk  # Minimise le risque

        def return_constraint(weights: np.ndarray) -> float:
            portfolio_return = PortfolioOptimizer.calculate_portfolio_performance(
                weights, daily_returns, 0
            ).return_value
            return portfolio_return - target_return  # Contrainte : rendement >= cible

        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            },  # Contrainte : somme des poids = 1
            {
                "type": "ineq",
                "fun": return_constraint,
            },  # Contrainte : rendement >= cible
        ]
        bounds = [(0.0, 1.0) for _ in range(num_assets)]  # Bornes : poids entre 0 et 1

        try:
            # Essaie plusieurs points de départ aléatoires si l'optimisation échoue
            for _ in range(5):
                result = minimize(
                    portfolio_risk,
                    PortfolioOptimizer.generate_random_weights(num_assets),
                    method="SLSQP",  # Méthode d'optimisation
                    constraints=constraints,
                    bounds=bounds,
                    options={"ftol": 1e-8, "maxiter": 1000},  # Options de précision
                )

                if result.success:
                    return result

            # Si toutes les tentatives échouent, essaie avec des contraintes assouplies
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {
                    "type": "ineq",
                    "fun": lambda w: return_constraint(w) + 0.01,
                },  # Ajoute une tolérance de 1%
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
                print(
                    f"Avertissement : L'optimisation du rendement cible a échoué : {result.message}"
                )

            return result

        except Exception as e:
            print(f"Erreur dans l'optimisation du rendement cible : {str(e)}")
            weights = (
                np.ones(num_assets) / num_assets
            )  # Retourne des poids égaux en cas d'erreur
            return OptimizeResult(
                x=weights,
                success=False,
                message=str(e),
                fun=portfolio_risk(weights),
            )


class PortfolioSimulator:
    """
    Simule et compare différentes stratégies de portefeuille.
    """

    def __init__(self, config: PortfolioConfig):
        """
        Initialise le simulateur avec la configuration.
        Args:
            config (PortfolioConfig): Configuration du portefeuille.
        """
        self.config = config
        self.data_loader = DataLoader()
        self.optimizer = PortfolioOptimizer()
        self._initialize_storage()

    def _initialize_storage(self):
        """Initialise les conteneurs de stockage des données."""
        self.dates: List[datetime] = []  # Dates des simulations
        self.portfolio_values: Dict[str, List[float]] = {
            strategy: []
            for strategy in ["sharpe", "equal_weight", "min_risk", "target_return"]
        }  # Valeurs du portefeuille par stratégie
        self.weights: Dict[str, List[np.ndarray]] = {
            strategy: []
            for strategy in ["sharpe", "equal_weight", "min_risk", "target_return"]
        }  # Poids des actifs par stratégie
        self.closing_prices_history: List[np.ndarray] = (
            []
        )  # Historique des prix de clôture
        self.performance_history: Dict[str, List[PortfolioPerformance]] = {
            strategy: []
            for strategy in ["sharpe", "equal_weight", "min_risk", "target_return"]
        }  # Historique des performances par stratégie

    def get_date_range(self, current_month: int) -> Tuple[datetime, datetime]:
        """
        Calcule les dates de début et de fin pour l'itération actuelle.
        Args:
            current_month (int): Mois actuel de la simulation.
        Returns:
            Tuple[datetime, datetime]: Dates de début et de fin.
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
        Calcule les poids pour toutes les stratégies de portefeuille.
        Args:
            daily_returns (pd.DataFrame): Rendements quotidiens.
            risk_free_rate (float): Taux sans risque.
        Returns:
            Dict[str, np.ndarray]: Poids par stratégie.
        """
        num_assets = len(self.config.tickers)

        weights = {
            "sharpe": self.optimizer.optimize_max_sharpe_ratio(
                daily_returns, num_assets, risk_free_rate
            ).x,  # Poids pour maximiser le ratio de Sharpe
            "equal_weight": np.ones(num_assets) / num_assets,  # Poids égaux
            "min_risk": self.optimizer.optimize_min_risk(
                daily_returns, num_assets
            ).x,  # Poids pour minimiser le risque
        }

        if self.config.target_return is not None:
            target_result = self.optimizer.optimize_for_target_return(
                daily_returns, num_assets, self.config.target_return
            )
            weights["target_return"] = (
                target_result.x
            )  # Poids pour atteindre le rendement cible

        return weights

    def _calculate_portfolio_value(
        self, month: int, strategy: str, weights: np.ndarray, current_prices: np.ndarray
    ) -> float:
        """
        Calcule la valeur du portefeuille pour une stratégie donnée.
        Args:
            month (int): Mois actuel de la simulation.
            strategy (str): Stratégie de portefeuille.
            weights (np.ndarray): Poids des actifs.
            current_prices (np.ndarray): Prix actuels des actifs.
        Returns:
            float: Valeur du portefeuille.
        """
        if month == 0:
            shares = (
                self.config.initial_capital * weights / current_prices
            )  # Nombre d'actions initial
            return np.sum(shares * current_prices)  # Valeur initiale du portefeuille
        else:
            prev_value = self.portfolio_values[strategy][month - 1]  # Valeur précédente
            prev_weights = self.weights[strategy][month - 1]  # Poids précédents
            prev_prices = self.closing_prices_history[month - 1]  # Prix précédents
            shares = prev_value * prev_weights / prev_prices  # Nombre d'actions
            return np.sum(shares * current_prices)  # Valeur actuelle du portefeuille

    def update_portfolio_values(
        self,
        month: int,
        weights: Dict[str, np.ndarray],
        current_prices: np.ndarray,
        daily_returns: pd.DataFrame,
        risk_free_rate: float,
    ):
        """
        Met à jour les valeurs et les métriques de performance du portefeuille pour toutes les stratégies.
        Args:
            month (int): Mois actuel de la simulation.
            weights (Dict[str, np.ndarray]): Poids par stratégie.
            current_prices (np.ndarray): Prix actuels des actifs.
            daily_returns (pd.DataFrame): Rendements quotidiens.
            risk_free_rate (float): Taux sans risque.
        """
        for strategy in self.portfolio_values.keys():
            if strategy == "target_return" and self.config.target_return is None:
                continue

            portfolio_value = self._calculate_portfolio_value(
                month, strategy, weights[strategy], current_prices
            )
            self.portfolio_values[strategy].append(
                portfolio_value
            )  # Ajoute la valeur actuelle
            self.weights[strategy].append(weights[strategy])  # Ajoute les poids actuels

            performance = self.optimizer.calculate_portfolio_performance(
                weights[strategy], daily_returns, risk_free_rate
            )
            self.performance_history[strategy].append(
                performance
            )  # Ajoute les métriques de performance

    def run_simulation(self):
        """Exécute la simulation du portefeuille."""
        for month in range(self.config.total_months):
            start_date, end_date = self.get_date_range(month)
            closing_prices = self.data_loader.load_data(
                self.config.tickers,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )  # Charge les prix de clôture

            daily_returns = self.data_loader.calculate_daily_returns(
                closing_prices
            )  # Calcule les rendements
            risk_free_rate = self.data_loader.get_risk_free_rate(
                start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )  # Récupère le taux sans risque

            weights = self.calculate_portfolio_weights(
                daily_returns, risk_free_rate
            )  # Calcule les poids
            current_prices = closing_prices.iloc[-1].values  # Prix actuels

            self.dates.append(end_date)  # Ajoute la date actuelle
            self.closing_prices_history.append(
                current_prices
            )  # Ajoute les prix actuels
            self.update_portfolio_values(
                month, weights, current_prices, daily_returns, risk_free_rate
            )  # Met à jour les valeurs du portefeuille

    def plot_results(self):
        """Trace les valeurs du portefeuille au fil du temps pour toutes les stratégies."""
        plt.figure(figsize=(12, 6))
        dates = pd.to_datetime(self.dates)
        styles = {
            "sharpe": "-",
            "equal_weight": "--",
            "min_risk": "-.",
            "target_return": ":",
        }  # Styles de ligne pour chaque stratégie

        for strategy, values in self.portfolio_values.items():
            if strategy == "target_return" and self.config.target_return is None:
                continue
            plt.plot(
                dates,
                values,
                label=strategy.replace("_", " ").title(),
                linestyle=styles[strategy],
            )  # Trace la courbe pour chaque stratégie

        plt.title("Comparaison des Valeurs du Portefeuille au Fil du Temps")
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
            Dict[str, Dict[str, float]]: Statistiques récapitulatives par stratégie.
        """
        summary = {}
        for strategy in self.portfolio_values.keys():
            if strategy == "target_return" and self.config.target_return is None:
                continue

            values = np.array(self.portfolio_values[strategy])
            monthly_returns = np.diff(values) / values[:-1]  # Rendements mensuels

            # Calcule les métriques de drawdown
            peak = np.maximum.accumulate(values)
            drawdown = (peak - values) / peak
            max_drawdown = np.max(drawdown)

            # Calcule les métriques de rendement
            total_years = self.config.total_months / 12
            total_return = (values[-1] / values[0]) - 1
            annualized_return = (1 + total_return) ** (1 / total_years) - 1

            # Calcule les métriques de risque
            monthly_volatility = np.std(monthly_returns)
            annualized_volatility = monthly_volatility * ANNUALIZATION_FACTOR

            summary[strategy] = {
                "final_value": values[-1],  # Valeur finale du portefeuille
                "total_return": total_return,  # Rendement total
                "annualized_return": annualized_return,  # Rendement annualisé
                "annualized_volatility": annualized_volatility,  # Volatilité annualisée
                "max_drawdown": max_drawdown,  # Drawdown maximum
                "sharpe_ratio": (
                    annualized_return / annualized_volatility
                    if annualized_volatility > 0
                    else 0
                ),  # Ratio de Sharpe
                "calmar_ratio": (
                    annualized_return / max_drawdown if max_drawdown > 0 else 0
                ),  # Ratio de Calmar
            }

        return summary
