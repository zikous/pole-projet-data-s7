from flask import Blueprint, jsonify, request  # Importe les modules Flask nécessaires
from services.market_service import (
    get_correlation_and_clusters,
)  # Importe la fonction d'analyse de corrélation
from services.stock_service import (
    get_stock_data,
)  # Importe la fonction pour récupérer les données boursières
from services.portfolio_service import (
    run_optimization,
    run_backtest,
)  # Importe les fonctions d'optimisation et de backtest
from services.chart_service import (
    create_portfolio_chart,
)  # Importe la fonction pour créer des graphiques de portefeuille
from datetime import datetime  # Importe datetime pour gérer les dates
import logging  # Importe logging pour la journalisation des erreurs

# Initialise le Blueprint pour les routes API
api_bp = Blueprint("api", __name__)

# Configure la journalisation
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@api_bp.route("/stock-info/<ticker>")
def get_stock_info(ticker):
    """
    Route pour récupérer les informations sur une action spécifique.
    Args:
        ticker (str): Symbole boursier (ex: "AAPL").
    Returns:
        JSON: Informations de base, graphique des prix et actualités.
    """
    try:
        # Récupère les données de l'action
        basic_info, hist_data, price_chart, news = get_stock_data(ticker)
        return jsonify(
            {
                "success": True,
                "info": basic_info,  # Informations de base
                "chart": price_chart,  # Graphique des prix
                "news": news,  # Actualités
            }
        )
    except Exception as e:
        logger.error(
            f"Erreur lors de la récupération des informations pour {ticker} : {e}"
        )
        return jsonify(
            {"success": False, "error": str(e)}
        )  # Retourne une erreur en cas d'échec


@api_bp.route("/optimize-portfolio", methods=["POST"])
def optimize_portfolio():
    """
    Route pour optimiser un portefeuille en fonction des paramètres fournis.
    Returns:
        JSON: Résultats de l'optimisation (poids, performances, etc.).
    """
    try:
        data = request.json  # Récupère les données de la requête
        logger.debug(f"Données reçues : {data}")

        # Valide les champs obligatoires
        required_fields = ["tickers", "initial_capital", "start_date", "end_date"]
        for field in required_fields:
            if field not in data:
                return jsonify(
                    {"success": False, "error": f"Champ obligatoire manquant : {field}"}
                )

        # Vérifie que les tickers sont une liste non vide
        if not isinstance(data["tickers"], list) or len(data["tickers"]) == 0:
            return jsonify(
                {
                    "success": False,
                    "error": "Les tickers doivent être une liste non vide",
                }
            )

        # Exécute l'optimisation du portefeuille
        result = run_optimization(
            data["tickers"],
            float(data["initial_capital"]),  # Capital initial
            datetime.strptime(data["start_date"], "%Y-%m-%d"),  # Date de début
            datetime.strptime(data["end_date"], "%Y-%m-%d"),  # Date de fin
            (
                float(data["target_return"]) if data.get("target_return") else None
            ),  # Rendement cible (optionnel)
        )

        logger.debug(f"Résultat de l'optimisation : {result}")

        # Vérifie si l'optimisation a retourné des résultats
        if not result:
            return jsonify(
                {
                    "success": False,
                    "error": "L'optimisation n'a retourné aucun résultat",
                }
            )

        # Prépare la réponse
        response = {
            "success": True,
            "result": {
                "weights": result.get("weights", {}),  # Poids des actifs
                "performance": result.get("performance", {}),  # Performances
                "tickers": result.get("tickers", data["tickers"]),  # Tickers
            },
        }

        logger.debug(f"Envoi de la réponse : {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Erreur lors de l'optimisation du portefeuille : {e}")
        return jsonify(
            {"success": False, "error": str(e)}
        )  # Retourne une erreur en cas d'échec


@api_bp.route("/backtest", methods=["POST"])
def backtest():
    """
    Route pour exécuter un backtest de portefeuille.
    Returns:
        JSON: Statistiques récapitulatives et graphique de performance.
    """
    try:
        data = request.json  # Récupère les données de la requête
        summary, chart = run_backtest(data)  # Exécute le backtest
        return jsonify({"success": True, "summary": summary, "chart": chart})
    except Exception as e:
        return jsonify(
            {"success": False, "error": str(e)}
        )  # Retourne une erreur en cas d'échec


@api_bp.route("/market-correlation", methods=["POST"])
def get_market_correlation():
    """
    Route pour analyser la corrélation et les clusters d'un ensemble d'actions.
    Returns:
        JSON: Matrice de corrélation, clusters et graphiques.
    """
    try:
        data = request.json  # Récupère les données de la requête
        logger.debug(f"Données de corrélation reçues : {data}")

        # Valide les tickers
        if not data.get("tickers") or not isinstance(data["tickers"], list):
            return jsonify({"success": False, "error": "Tickers invalides"})

        # Valide les dates
        if not data.get("start_date") or not data.get("end_date"):
            return jsonify(
                {
                    "success": False,
                    "error": "Les dates de début et de fin sont obligatoires",
                }
            )

        # Effectue l'analyse de corrélation et de clustering
        result = get_correlation_and_clusters(
            data["tickers"], data["start_date"], data["end_date"]
        )

        # Vérifie si l'analyse a réussi
        if not result.get("success"):
            return jsonify(result)

        # Retourne les résultats de l'analyse
        return jsonify(result)

    except Exception as e:
        logger.error(
            f"Erreur lors de l'analyse de corrélation : {str(e)}", exc_info=True
        )
        return jsonify(
            {"success": False, "error": str(e)}
        )  # Retourne une erreur en cas d'échec
