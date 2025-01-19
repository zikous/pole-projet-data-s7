# Importation des modules nécessaires
import numpy as np  # Pour les calculs numériques
from scipy.optimize import minimize, OptimizeResult  # Pour l'optimisation
from ..config.settings import TRADING_DAYS_PER_YEAR  # Nombre de jours de trading par an
from .performance import (
    PortfolioPerformance,
)  # Classe pour stocker les métriques de performance
import pandas as pd  # Pour manipuler les données sous forme de DataFrame


class PortfolioOptimizer:
    """
    Classe responsable de l'optimisation de portefeuille.
    Elle fournit des méthodes pour :
    - Générer des poids aléatoires pour les actifs.
    - Calculer les métriques de performance d'un portefeuille.
    - Optimiser le portefeuille pour maximiser le ratio de Sharpe, minimiser le risque, ou atteindre un rendement cible.
    """

    @staticmethod
    def generate_random_weights(num_assets: int) -> np.ndarray:
        """
        Génère des poids aléatoires pour les actifs du portefeuille.
        Les poids sont générés de manière à ce qu'ils somment à 1.

        Args:
            num_assets (int) : Nombre d'actifs dans le portefeuille.

        Returns:
            np.ndarray : Un tableau de poids aléatoires.
        """
        return np.random.dirichlet(np.ones(num_assets))

    @staticmethod
    def calculate_portfolio_performance(
        weights: np.ndarray, daily_returns: pd.DataFrame, risk_free_rate: float
    ) -> PortfolioPerformance:
        """
        Calcule les métriques de performance du portefeuille : rendement, risque et ratio de Sharpe.

        Args:
            weights (np.ndarray) : Les poids des actifs dans le portefeuille.
            daily_returns (pd.DataFrame) : Les rendements quotidiens des actifs.
            risk_free_rate (float) : Le taux sans risque.

        Returns:
            PortfolioPerformance : Un objet contenant le rendement, le risque et le ratio de Sharpe.
        """
        # Calcul des métriques annualisées
        annualized_returns = (1 + daily_returns.mean()) ** TRADING_DAYS_PER_YEAR - 1
        annualized_covariance = daily_returns.cov() * TRADING_DAYS_PER_YEAR

        # Calcul du rendement et du risque du portefeuille
        portfolio_return = np.sum(annualized_returns * weights)
        portfolio_risk = np.sqrt(
            np.dot(weights.T, np.dot(annualized_covariance, weights))
        )

        # Calcul du ratio de Sharpe (avec vérification pour éviter la division par zéro)
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
            daily_returns (pd.DataFrame) : Les rendements quotidiens des actifs.
            num_assets (int) : Nombre d'actifs dans le portefeuille.
            risk_free_rate (float) : Le taux sans risque.

        Returns:
            OptimizeResult : Résultat de l'optimisation.
        """

        def negative_sharpe_ratio(weights: np.ndarray) -> float:
            # Calcule le ratio de Sharpe négatif (pour la minimisation)
            perf = PortfolioOptimizer.calculate_portfolio_performance(
                weights, daily_returns, risk_free_rate
            )
            return -perf.sharpe_ratio

        # Contraintes : les poids doivent sommer à 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        # Bornes : chaque poids doit être entre 0 et 1
        bounds = [(0.0, 1.0) for _ in range(num_assets)]

        # Optimisation avec la méthode SLSQP
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
        """
        Optimise les poids du portefeuille pour minimiser le risque.

        Args:
            daily_returns (pd.DataFrame) : Les rendements quotidiens des actifs.
            num_assets (int) : Nombre d'actifs dans le portefeuille.

        Returns:
            OptimizeResult : Résultat de l'optimisation.
        """

        def portfolio_risk(weights: np.ndarray) -> float:
            # Calcule le risque du portefeuille
            return PortfolioOptimizer.calculate_portfolio_performance(
                weights, daily_returns, 0
            ).risk

        # Contraintes : les poids doivent sommer à 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        # Bornes : chaque poids doit être entre 0 et 1
        bounds = [(0.0, 1.0) for _ in range(num_assets)]

        # Optimisation avec la méthode SLSQP
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
        Optimise les poids du portefeuille pour minimiser le risque tout en atteignant un rendement cible.

        Args:
            daily_returns (pd.DataFrame) : Les rendements quotidiens des actifs.
            num_assets (int) : Nombre d'actifs dans le portefeuille.
            target_return (float) : Rendement annuel cible (en décimal).

        Returns:
            OptimizeResult : Résultat de l'optimisation.
        """

        def portfolio_risk(weights: np.ndarray) -> float:
            # Calcule le risque du portefeuille
            return PortfolioOptimizer.calculate_portfolio_performance(
                weights, daily_returns, 0
            ).risk

        def return_constraint(weights: np.ndarray) -> float:
            # Calcule le rendement du portefeuille et vérifie s'il atteint le rendement cible
            portfolio_return = PortfolioOptimizer.calculate_portfolio_performance(
                weights, daily_returns, 0
            ).return_value
            return portfolio_return - target_return

        # Contraintes : les poids doivent sommer à 1, et le rendement doit atteindre la cible
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Somme des poids = 1
            {"type": "ineq", "fun": return_constraint},  # Rendement >= cible
        ]
        # Bornes : chaque poids doit être entre 0 et 1
        bounds = [(0.0, 1.0) for _ in range(num_assets)]

        try:
            # Essayer plusieurs points de départ aléatoires si l'optimisation échoue
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

            # Si toutes les tentatives échouent, relâcher les contraintes
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {
                    "type": "ineq",
                    "fun": lambda w: return_constraint(w) + 0.01,
                },  # Ajouter une tolérance de 1%
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
                    f"Warning: L'optimisation du rendement cible a échoué : {result.message}"
                )

            return result

        except Exception as e:
            # En cas d'erreur, retourner un résultat par défaut
            print(f"Erreur lors de l'optimisation du rendement cible : {str(e)}")
            weights = np.ones(num_assets) / num_assets  # Poids égaux
            return OptimizeResult(
                x=weights,
                success=False,
                message=str(e),
                fun=portfolio_risk(weights),
            )
