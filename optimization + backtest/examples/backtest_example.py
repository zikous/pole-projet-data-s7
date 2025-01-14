import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.config.settings import PortfolioConfig
from src.simulation.simulator import PortfolioSimulator


def main():
    """Example usage of the portfolio optimization system"""
    np.random.seed(42)  # For reproducibility

    config = PortfolioConfig(
        initial_capital=100000,
        lookback_months=6,
        total_months=12 * 3,
        start_year=2021,
        tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD"],
        target_return=0.15,
    )
    # config = PortfolioConfig(
    #     initial_capital=100000,  # Capital initial de $100,000
    #     lookback_months=12,  # 1 an de données historiques pour avoir une meilleure perspective
    #     total_months=12 * 4,  # 4 ans pour capturer avant/pendant/après la crise
    #     start_year=2007,  # Commence avant la crise pour voir l'impact complet
    #     tickers=[  # Mix d'actifs fortement impactés par la crise
    #         "C",  # Citigroup (banque fortement touchée)
    #         "BAC",  # Bank of America (secteur financier)
    #         "AIG",  # AIG (assurance, quasi-faillite)
    #         "GE",  # General Electric (industriel cyclique)
    #         "XOM",  # Exxon Mobil (énergie)
    #         "SPY",  # S&P 500 ETF (marché général)
    #         "GLD",  # Gold ETF (valeur refuge)
    #         "TLT",  # Long-Term Treasury ETF (obligations gouvernementales)
    #     ],
    #     target_return=0.40,  # Target return
    # )
    # config = PortfolioConfig(
    #     initial_capital=100000,  # Capital initial de $100,000
    #     lookback_months=12,  # 1 an de données historiques pour avoir une meilleure perspective
    #     total_months=12 * 4,  # 4 ans pour capturer avant/pendant/après la crise
    #     start_year=2007,  # Commence avant la crise pour voir l'impact complet
    #     tickers=[  # Mix d'actifs plus volatils pendant la crise financière
    #         "F",  # Ford (automobile, volatile pendant la crise)
    #         "C",  # Citigroup (banque fortement touchée)
    #         "BAC",  # Bank of America (secteur financier)
    #         "AIG",  # AIG (assurance, quasi-faillite)
    #         "XLF",  # Financial Select Sector SPDR Fund (secteur financier)
    #     ],
    #     target_return=0.30,  # Target return
    # )
    # config = PortfolioConfig(
    #     initial_capital=100000,  # Capital initial de $100,000. Cela représente le montant avec lequel nous commençons à investir.
    #     lookback_months=12,  # Période d'observation de 12 mois, pour observer la performance des actifs sur une année normale.
    #     total_months=12
    #     * 4,  # 4 ans d'analyse pour capturer un cycle économique complet sans les effets perturbateurs de crises majeures.
    #     start_year=2012,  # Année de début. Nous commençons après la crise financière pour analyser une période de reprise économique.
    #     tickers=[  # Liste des actifs choisie. Ces actifs sont typiquement moins volatils, représentant une période de croissance stable.
    #         "AAPL",  # Apple Inc. (symbolisé par "AAPL") - L'une des entreprises les plus stables et performantes dans la période post-crise.
    #         "MSFT",  # Microsoft Corporation (symbolisé par "MSFT") - Une entreprise technologique ayant connu une croissance stable et régulière.
    #         "GOOGL",  # Alphabet Inc. (symbolisé par "GOOGL") - La société mère de Google, une autre entreprise de technologie dominante sur le marché.
    #         "V",  # Visa Inc. (symbolisé par "V") - Une société de paiement stable et largement répandue, généralement moins affectée par des chocs économiques.
    #         "SPY",  # S&P 500 ETF (symbolisé par "SPY") - Un fonds indiciel qui suit les 500 plus grandes entreprises cotées en bourse, représentant un portefeuille équilibré du marché américain.
    #     ],
    #     target_return=0.10,  # Rendement cible de 10%. Un objectif de rendement plus modéré et réaliste pour un environnement économique stable.
    # )

    try:
        simulator = PortfolioSimulator(config)
        simulator.run_simulation()
        simulator.plot_results()

        # Display results
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
        print(f"Error in portfolio simulation: {str(e)}")


if __name__ == "__main__":
    main()
