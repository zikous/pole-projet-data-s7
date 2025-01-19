from services.utils import (
    PortfolioConfig,  # Importe la classe pour configurer le portefeuille
    PortfolioSimulator,  # Importe la classe pour simuler le portefeuille
    DataLoader,  # Importe la classe pour charger les données
    PortfolioOptimizer,  # Importe la classe pour optimiser le portefeuille
)
import numpy as np  # Importe numpy pour les calculs numériques
from services.chart_service import (
    create_portfolio_chart,
)  # Importe la fonction pour créer des graphiques
import logging  # Importe logging pour la journalisation des erreurs


def run_optimization(
    tickers, initial_capital, start_date, end_date, target_return=None
):
    """
    Exécute l'optimisation du portefeuille pour une liste de tickers donnée.
    Args:
        tickers (list): Liste des symboles boursiers à inclure dans le portefeuille.
        initial_capital (float): Capital initial pour le portefeuille.
        start_date (datetime): Date de début pour les données historiques.
        end_date (datetime): Date de fin pour les données historiques.
        target_return (float, optional): Rendement cible pour l'optimisation. Par défaut, None.
    Returns:
        dict: Résultats de l'optimisation (poids, performances, etc.).
    """
    # Initialise un logger pour le débogage
    logger = logging.getLogger(__name__)

    # Log les paramètres de la fonction pour le débogage
    logger.debug(
        f"""
    Exécution de l'optimisation avec :
    - tickers: {tickers}
    - capital initial: {initial_capital}
    - date de début: {start_date}
    - date de fin: {end_date}
    - rendement cible: {target_return}
    """
    )

    try:
        # Étape 1 : Charge les données de prix des actions
        data_loader = DataLoader()  # Instancie un chargeur de données
        prices = data_loader.load_data(
            tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )  # Charge les données de prix pour les tickers et la période spécifiée

        # Vérifie si les données de prix sont vides et lève une erreur si c'est le cas
        if prices.empty:
            raise ValueError(
                "Aucune donnée de prix trouvée pour les tickers et la période donnée"
            )

        # Étape 2 : Calcule les rendements quotidiens à partir des données de prix
        daily_returns = data_loader.calculate_daily_returns(
            prices
        )  # Calcule les rendements quotidiens
        risk_free_rate = data_loader.get_risk_free_rate(
            start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )  # Obtient le taux sans risque pour la période spécifiée

        # Étape 3 : Instancie un optimiseur de portefeuille
        optimizer = PortfolioOptimizer()
        num_assets = len(tickers)  # Obtient le nombre d'actifs (actions)

        # Initialise des dictionnaires pour stocker les poids et les performances des stratégies
        weights = {}
        performance = {}

        # Étape 4 : Effectue l'optimisation pour différentes stratégies

        # Maximise le ratio de Sharpe (optimise pour le meilleur rendement par unité de risque)
        sharpe_result = optimizer.optimize_max_sharpe_ratio(
            daily_returns, num_assets, risk_free_rate
        )
        if not hasattr(sharpe_result, "x"):
            raise ValueError("L'optimisation du ratio de Sharpe a échoué")
        weights["sharpe"] = (
            sharpe_result.x.tolist()
        )  # Stocke les poids pour la stratégie Sharpe

        # Portefeuille à pondération égale (attribue des poids égaux à chaque actif)
        equal_weights = np.ones(num_assets) / num_assets
        weights["equal_weight"] = equal_weights.tolist()

        # Minimise le risque (optimise pour le portefeuille avec la plus faible volatilité)
        min_risk_result = optimizer.optimize_min_risk(daily_returns, num_assets)
        if not hasattr(min_risk_result, "x"):
            raise ValueError("L'optimisation du risque minimum a échoué")
        weights["min_risk"] = (
            min_risk_result.x.tolist()
        )  # Stocke les poids pour la stratégie de risque minimum

        # Optimisation pour un rendement cible (optimise pour un rendement spécifique)
        if target_return is not None:
            target_result = optimizer.optimize_for_target_return(
                daily_returns, num_assets, target_return
            )
            if not hasattr(target_result, "x"):
                raise ValueError("L'optimisation du rendement cible a échoué")
            weights["target_return"] = (
                target_result.x.tolist()
            )  # Stocke les poids pour la stratégie de rendement cible

        # Étape 5 : Calcule les métriques de performance pour chaque stratégie
        for strategy, w in weights.items():
            w_array = np.array(w)  # Convertit les poids en un tableau numpy
            perf = optimizer.calculate_portfolio_performance(
                w_array, daily_returns, risk_free_rate
            )  # Calcule la performance du portefeuille (rendement, risque, ratio de Sharpe)
            performance[strategy] = {
                "return": float(perf.return_value),  # Rendement
                "risk": float(perf.risk),  # Risque
                "sharpe_ratio": float(perf.sharpe_ratio),  # Ratio de Sharpe
            }

        # Étape 6 : Compile les résultats de l'optimisation dans un dictionnaire
        result = {"weights": weights, "performance": performance, "tickers": tickers}

        # Log les résultats de l'optimisation pour le débogage
        logger.debug(f"Résultat de l'optimisation : {result}")

        # Retourne les résultats
        return result

    except Exception as e:
        # Log l'erreur si une exception se produit pendant l'optimisation
        logger.error(
            f"Erreur dans le calcul de l'optimisation : {str(e)}", exc_info=True
        )

        # Relance l'exception pour qu'elle puisse être gérée ailleurs
        raise


def run_backtest(data):
    """
    Exécute un backtest du portefeuille en fonction des paramètres fournis.
    Args:
        data (dict): Dictionnaire contenant les paramètres du backtest.
    Returns:
        tuple: Statistiques récapitulatives et graphique de performance du portefeuille.
    """
    # Étape 1 : Crée un objet PortfolioConfig avec les paramètres fournis
    config = PortfolioConfig(
        initial_capital=float(data["initial_capital"]),  # Capital initial
        lookback_months=int(data["lookback_months"]),  # Période de lookback en mois
        total_months=int(data["total_months"]),  # Période totale du backtest en mois
        start_year=int(data["start_year"]),  # Année de début du backtest
        tickers=data["tickers"],  # Liste des tickers
        target_return=(
            float(data["target_return"]) if data["target_return"] else None
        ),  # Rendement cible (optionnel)
    )

    # Étape 2 : Initialise le simulateur de portefeuille avec la configuration
    simulator = PortfolioSimulator(config)

    # Étape 3 : Exécute la simulation
    simulator.run_simulation()

    # Étape 4 : Génère un graphique de performance du portefeuille
    chart = create_portfolio_chart(simulator.portfolio_values, simulator.dates)

    # Étape 5 : Retourne les statistiques récapitulatives et le graphique
    return simulator.get_summary_statistics(), chart
