# Importation des modules nécessaires
import yfinance as yf  # Pour récupérer les données boursières
import pandas as pd  # Pour manipuler les données sous forme de DataFrame
from ..config.settings import DEFAULT_RISK_FREE_RATE  # Taux sans risque par défaut
from typing import List  # Pour le typage des listes


class DataLoader:
    """
    Classe responsable du chargement et du traitement des données boursières.
    Elle fournit des méthodes pour :
    - Charger les prix de clôture des actions.
    - Calculer les rendements quotidiens.
    - Récupérer le taux sans risque.
    """

    @staticmethod
    def load_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Charge les prix de clôture des actions pour une liste de tickers et une période donnée.

        Args:
            tickers (List[str]) : Liste des symboles boursiers (ex: ["AAPL", "MSFT"]).
            start_date (str) : Date de début au format 'AAAA-MM-JJ'.
            end_date (str) : Date de fin au format 'AAAA-MM-JJ'.

        Returns:
            pd.DataFrame : Un DataFrame contenant les prix de clôture des actions.

        Raises:
            ValueError : Si aucune donnée n'est récupérée ou en cas d'erreur.
        """
        try:
            # Téléchargement des données boursières via yfinance
            data = yf.download(tickers, start=start_date, end=end_date)

            # Vérification si les données sont vides
            if data.empty:
                raise ValueError("Aucune donnée récupérée pour la période spécifiée.")

            # Retourne uniquement les prix de clôture
            return data["Close"]

        except Exception as e:
            # Gestion des erreurs (ex: problème de connexion, tickers invalides)
            raise ValueError(f"Erreur lors du chargement des données : {str(e)}")

    @staticmethod
    def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les rendements quotidiens à partir des prix de clôture.

        Args:
            prices (pd.DataFrame) : DataFrame contenant les prix de clôture.

        Returns:
            pd.DataFrame : Un DataFrame contenant les rendements quotidiens.
        """
        # Calcul des rendements quotidiens en pourcentage et suppression des valeurs manquantes
        return prices.pct_change().dropna()

    @staticmethod
    def get_risk_free_rate(start_date: str, end_date: str) -> float:
        """
        Récupère le taux sans risque moyen à partir des rendements du Trésor américain (^TNX).
        Si les données ne sont pas disponibles, retourne un taux par défaut.

        Args:
            start_date (str) : Date de début au format 'AAAA-MM-JJ'.
            end_date (str) : Date de fin au format 'AAAA-MM-JJ'.

        Returns:
            float : Le taux sans risque moyen sous forme décimale (ex: 0.05 pour 5%).
        """
        try:
            # Téléchargement des données du Trésor américain (^TNX représente l'indice des taux d'intérêt)
            data = yf.download("^TNX", start=start_date, end=end_date)

            # Si les données sont vides, retourne le taux par défaut
            if data.empty:
                return DEFAULT_RISK_FREE_RATE

            # Calcul de la moyenne des taux de clôture et conversion en pourcentage
            # Note : Les taux sont donnés en pourcentage (ex: 5.0 pour 5%), donc on divise par 100
            return float(data["Close"].mean().iloc[0]) / 100

        except Exception as e:
            # En cas d'erreur (ex: problème de connexion, données indisponibles), retourne le taux par défaut
            print(f"Erreur lors de la récupération du taux sans risque : {e}")
            return DEFAULT_RISK_FREE_RATE
