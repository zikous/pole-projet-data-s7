from flask import Blueprint, render_template  # Importe les modules Flask nécessaires

# Initialise le Blueprint pour les routes principales
main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def home():
    """
    Route pour la page d'accueil.
    Returns:
        HTML: Page d'accueil (index.html).
    """
    return render_template("index.html")  # Affiche le template index.html


@main_bp.route("/stock-analysis")
def stock_analysis():
    """
    Route pour la page d'analyse d'actions.
    Returns:
        HTML: Page d'analyse d'actions (stock_analysis.html).
    """
    return render_template(
        "stock_analysis.html"
    )  # Affiche le template stock_analysis.html


@main_bp.route("/portfolio-optimizer")
def portfolio_optimizer():
    """
    Route pour la page d'optimisation de portefeuille.
    Returns:
        HTML: Page d'optimisation de portefeuille (portfolio_optimizer.html).
    """
    return render_template(
        "portfolio_optimizer.html"
    )  # Affiche le template portfolio_optimizer.html


@main_bp.route("/backtesting")
def backtesting():
    """
    Route pour la page de backtesting.
    Returns:
        HTML: Page de backtesting (backtesting.html).
    """
    return render_template("backtesting.html")  # Affiche le template backtesting.html


@main_bp.route("/market-analysis")
def market_analysis():
    """
    Route pour la page d'analyse de marché.
    Returns:
        HTML: Page d'analyse de marché (market_analysis.html).
    """
    return render_template(
        "market_analysis.html"
    )  # Affiche le template market_analysis.html
