# Importation des modules nécessaires
from typing import (
    List,
    Optional,
)  # Pour le typage des listes et des valeurs optionnelles
from dataclasses import dataclass  # Pour créer des classes de données simples
import numpy as np  # Pour les calculs numériques

# Constantes globales
TRADING_DAYS_PER_YEAR = 252  # Nombre de jours de trading dans une année
DEFAULT_RISK_FREE_RATE = 0.02  # Taux sans risque par défaut (2%)
ANNUALIZATION_FACTOR = np.sqrt(12)  # Facteur d'annualisation pour les calculs mensuels


@dataclass
class PortfolioConfig:
    """
    Configuration pour l'optimisation de portefeuille.
    Cette classe stocke les paramètres nécessaires pour définir un portefeuille et ses objectifs.

    Attributs :
        initial_capital (float) : Le capital initial investi dans le portefeuille.
        lookback_months (int) : Le nombre de mois de données historiques à utiliser pour l'analyse.
        total_months (int) : La durée totale de la simulation en mois.
        start_year (int) : L'année de début de la simulation.
        tickers (List[str]) : La liste des symboles boursiers (tickers) des actifs à inclure dans le portefeuille.
        target_return (Optional[float]) : Le rendement cible du portefeuille (optionnel). Si non spécifié, aucun rendement cible n'est fixé.
    """

    initial_capital: float  # Capital initial (ex: 100000 pour 100 000 $)
    lookback_months: int  # Nombre de mois de données historiques (ex: 12 pour 1 an)
    total_months: int  # Durée totale de la simulation (ex: 36 pour 3 ans)
    start_year: int  # Année de début (ex: 2020)
    tickers: List[str]  # Liste des tickers (ex: ["AAPL", "MSFT", "GOOGL"])
    target_return: Optional[float] = (
        None  # Rendement cible (ex: 0.10 pour 10%, optionnel)
    )
