from flask import Blueprint, render_template

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def home():
    return render_template("index.html")


@main_bp.route("/stock-analysis")
def stock_analysis():
    return render_template("stock_analysis.html")


@main_bp.route("/portfolio-optimizer")
def portfolio_optimizer():
    return render_template("portfolio_optimizer.html")


@main_bp.route("/backtesting")
def backtesting():
    return render_template("backtesting.html")


@main_bp.route("/market-analysis")
def market_analysis():
    return render_template("market_analysis.html")
