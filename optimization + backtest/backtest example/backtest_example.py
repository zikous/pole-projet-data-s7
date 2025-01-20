import sys
import os

# Ajout du chemin du projet au système pour permettre l'importation des modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.config.settings import PortfolioConfig  # Configuration du portefeuille
from src.simulation.simulator import PortfolioSimulator  # Simulateur de portefeuille


def main():
    """
    Exemple d'utilisation du système d'optimisation de portefeuille.
    Ce script configure un portefeuille, exécute une simulation et affiche les résultats.
    """
    np.random.seed(42)  # Pour assurer la reproductibilité des résultats

    # Exemples de configurations de portefeuille (commentées pour référence)
    # Chaque configuration représente un scénario différent (stabilité, crise, croissance, etc.)

    # Configuration 1 : Portefeuille tech moderne (post-crise financière)
    config = PortfolioConfig(
        initial_capital=100000,  # Capital initial de $100,000
        lookback_months=6,  # 6 mois de données historiques
        total_months=12 * 4,  # 3 ans de simulation
        start_year=2021,  # Début en 2021
        tickers=[
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "AMD",
        ],  # Actifs tech
        target_return=0.15,  # Rendement cible de 15%
    )

    # Configuration 2 : Portefeuille pendant la crise financière de 2008
    # config = PortfolioConfig(
    #     initial_capital=100000,  # Capital initial de $100,000
    #     lookback_months=12,  # 1 an de données historiques
    #     total_months=12 * 4,  # 4 ans pour capturer avant/pendant/après la crise
    #     start_year=2007,  # Début avant la crise
    #     tickers=[  # Actifs fortement impactés par la crise
    #         "C",  # Citigroup (banque)
    #         "BAC",  # Bank of America
    #         "AIG",  # AIG (assurance)
    #         "GE",  # General Electric
    #         "XOM",  # Exxon Mobil (énergie)
    #         "SPY",  # S&P 500 ETF (marché général)
    #         "GLD",  # Gold ETF (valeur refuge)
    #         "TLT",  # Long-Term Treasury ETF (obligations gouvernementales)
    #     ],
    #     target_return=0.40,  # Rendement cible ambitieux de 40%
    # )

    # Configuration 3 : Portefeuille volatil pendant la crise financière
    # config = PortfolioConfig(
    #     initial_capital=100000,  # Capital initial de $100,000
    #     lookback_months=12,  # 1 an de données historiques
    #     total_months=12 * 4,  # 4 ans de simulation
    #     start_year=2007,  # Début avant la crise
    #     tickers=[  # Actifs très volatils pendant la crise
    #         "F",  # Ford (automobile)
    #         "C",  # Citigroup (banque)
    #         "BAC",  # Bank of America
    #         "AIG",  # AIG (assurance)
    #         "XLF",  # Financial Select Sector SPDR Fund (secteur financier)
    #     ],
    #     target_return=0.40,  # Rendement cible de 40%
    # )

    # Configuration 4 : Portefeuille stable post-crise (2012-2016)
    # config = PortfolioConfig(
    #     initial_capital=100000,  # Capital initial de $100,000
    #     lookback_months=12,  # 1 an de données historiques
    #     total_months=12 * 4,  # 4 ans de simulation
    #     start_year=2012,  # Début après la crise
    #     tickers=[  # Actifs stables et performants
    #         "AAPL",  # Apple Inc.
    #         "MSFT",  # Microsoft Corporation
    #         "GOOGL",  # Alphabet Inc.
    #         "V",  # Visa Inc.
    #         "SPY",  # S&P 500 ETF
    #     ],
    #     target_return=0.10,  # Rendement cible modéré de 10%
    # )

    # # Configuration 5 : Portefeuille de défense (2000-2005, période de conflits)
    # config = PortfolioConfig(
    #     initial_capital=100000,  # Capital initial de $100,000
    #     lookback_months=12,  # 1 an de données historiques
    #     total_months=12 * 5,  # 5 ans de simulation
    #     start_year=2000,  # Début avant la guerre en Irak
    #     tickers=[  # Entreprises de défense et d'armement
    #         "LMT",  # Lockheed Martin
    #         "BA",  # Boeing
    #         "RTX",  # Raytheon Technologies
    #         "GD",  # General Dynamics
    #         "NOC",  # Northrop Grumman
    #     ],
    #     target_return=0.10,  # Rendement cible de 10%
    # )

    try:
        # Initialisation du simulateur avec la configuration choisie
        simulator = PortfolioSimulator(config)
        # Exécution de la simulation
        simulator.run_simulation()
        # Affichage des résultats graphiques
        simulator.plot_results()

        # Affichage des statistiques récapitulatives
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
        # Gestion des erreurs pendant la simulation
        print(f"Error in portfolio simulation: {str(e)}")


if __name__ == "__main__":
    # Point d'entrée du script
    main()
