# Importation des modules nécessaires
from dataclasses import dataclass  # Pour créer des classes de données simples
from typing import Dict  # Pour le typage des dictionnaires


@dataclass
class PortfolioPerformance:
    """
    Stocke les métriques de performance d'un portefeuille.
    Cette classe est utilisée pour encapsuler les résultats de performance d'un portefeuille,
    tels que le rendement, le risque et le ratio de Sharpe.

    Attributs :
        return_value (float) : Le rendement du portefeuille (ex: 0.10 pour 10%).
        risk (float) : Le risque du portefeuille, généralement mesuré par l'écart-type des rendements.
        sharpe_ratio (float) : Le ratio de Sharpe, qui mesure le rendement ajusté au risque.

    Méthodes :
        to_dict() : Convertit les métriques de performance en un dictionnaire.
    """

    return_value: float  # Rendement du portefeuille
    risk: float  # Risque du portefeuille (écart-type des rendements)
    sharpe_ratio: float  # Ratio de Sharpe (rendement ajusté au risque)

    def to_dict(self) -> Dict[str, float]:
        """
        Convertit les métriques de performance en un dictionnaire.

        Returns:
            Dict[str, float] : Un dictionnaire contenant les métriques de performance :
                - "return" : Le rendement du portefeuille.
                - "risk" : Le risque du portefeuille.
                - "sharpe_ratio" : Le ratio de Sharpe.
        """
        return {
            "return": self.return_value,
            "risk": self.risk,
            "sharpe_ratio": self.sharpe_ratio,
        }
