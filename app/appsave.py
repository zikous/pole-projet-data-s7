# app.py
from flask import Flask, render_template, request, jsonify
from ok import (
    PortfolioConfig,
    PortfolioSimulator,
    DataLoader,
    PortfolioOptimizer,
    TRADING_DAYS_PER_YEAR,
)
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from dateutil.relativedelta import relativedelta
import json
import plotly

app = Flask(__name__)


def get_stock_data(ticker):
    """Get stock information, historical data, and news"""
    stock = yf.Ticker(ticker)
    info = stock.info

    # Basic Stock Information
    basic_info = {
        "name": info.get("longName", "N/A"),
        "sector": info.get("sector", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "pe_ratio": info.get("trailingPE", "N/A"),
        "dividend_yield": info.get("dividendYield", "N/A"),
        "beta": info.get("beta", "N/A"),
        "profit_margin": info.get("profitMargins", "N/A"),
        "debt_to_equity": info.get("debtToEquity", "N/A"),
        "roe": info.get("returnOnEquity", "N/A"),
        "roa": info.get("returnOnAssets", "N/A"),
    }

    # Historical Data with Technical Indicators
    hist = stock.history(period="1y")
    hist["SMA_50"] = hist["Close"].rolling(window=50).mean()
    hist["SMA_200"] = hist["Close"].rolling(window=200).mean()

    price_chart = create_price_chart(hist)

    # News Data (Example structure, adjust as per your API availability)
    news = []
    try:
        stock_news = stock.news[:5]  # Fetch latest 5 news articles
        news = [{"title": item["title"], "link": item["link"]} for item in stock_news]
    except Exception:
        news = []

    return basic_info, hist, price_chart, news


def create_price_chart(hist_data):
    """Create an interactive price chart using plotly"""
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=hist_data.index,
                open=hist_data["Open"],
                high=hist_data["High"],
                low=hist_data["Low"],
                close=hist_data["Close"],
            )
        ]
    )

    fig.update_layout(
        title="Price History",
        yaxis_title="Price",
        xaxis_title="Date",
        template="plotly_dark",
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def run_optimization(
    tickers, initial_capital, start_date, end_date, target_return=None
):
    """Run portfolio optimization for the web interface"""
    config = PortfolioConfig(
        initial_capital=float(initial_capital),
        lookback_months=6,
        total_months=12,
        start_year=start_date.year,
        tickers=tickers,
        target_return=target_return,
    )

    simulator = PortfolioSimulator(config)
    simulator.run_simulation()

    return simulator.get_summary_statistics(), simulator.weights


def create_portfolio_chart(values, dates):
    """Create an interactive portfolio performance chart"""
    fig = go.Figure()

    for strategy, performance in values.items():
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=performance,
                name=strategy.replace("_", " ").title(),
                mode="lines",
            )
        )

    fig.update_layout(
        title="Portfolio Performance Comparison",
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Date",
        template="plotly_dark",
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/stock-analysis")
def stock_analysis():
    return render_template("stock_analysis.html")


@app.route("/api/stock-info/<ticker>")
def get_stock_info(ticker):
    try:
        basic_info, hist_data, price_chart, news = get_stock_data(ticker)
        return jsonify(
            {"success": True, "info": basic_info, "chart": price_chart, "news": news}
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/portfolio-optimizer")
def portfolio_optimizer():
    return render_template("portfolio_optimizer.html")


# Add this updated route to your app.py


@app.route("/api/optimize-portfolio", methods=["POST"])
def optimize_portfolio():
    try:
        data = request.json
        print("Received data:", data)  # Debug print

        # Validate input data
        if not data.get("tickers") or not data.get("initial_capital"):
            raise ValueError("Missing required parameters")

        tickers = data["tickers"]
        initial_capital = float(data["initial_capital"])
        start_date = datetime.strptime(data["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(data["end_date"], "%Y-%m-%d")

        # Convert target return to decimal if provided
        target_return = (
            float(data["target_return"]) if data.get("target_return") else None
        )

        # Initialize DataLoader and get historical data
        data_loader = DataLoader()
        prices = data_loader.load_data(
            tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )

        # Calculate daily returns
        daily_returns = data_loader.calculate_daily_returns(prices)

        # Get risk-free rate
        risk_free_rate = data_loader.get_risk_free_rate(
            start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )

        # Initialize optimizer
        optimizer = PortfolioOptimizer()
        num_assets = len(tickers)

        # Calculate optimal weights for different strategies
        results = {}
        weights = {}

        # Max Sharpe Ratio strategy
        sharpe_result = optimizer.optimize_max_sharpe_ratio(
            daily_returns, num_assets, risk_free_rate
        )
        weights["sharpe"] = sharpe_result.x

        # Equal Weight strategy
        weights["equal_weight"] = np.ones(num_assets) / num_assets

        # Minimum Risk strategy
        min_risk_result = optimizer.optimize_min_risk(daily_returns, num_assets)
        weights["min_risk"] = min_risk_result.x

        # Target Return strategy (if specified)
        if target_return is not None:
            target_result = optimizer.optimize_for_target_return(
                daily_returns, num_assets, target_return
            )
            weights["target_return"] = target_result.x

        # Calculate performance metrics for each strategy
        performance = {}
        for strategy, w in weights.items():
            perf = optimizer.calculate_portfolio_performance(
                w, daily_returns, risk_free_rate
            )
            performance[strategy] = {
                "return": float(perf.return_value),
                "risk": float(perf.risk),
                "sharpe_ratio": float(perf.sharpe_ratio),
            }

        return jsonify(
            {
                "success": True,
                "weights": {k: v.tolist() for k, v in weights.items()},
                "summary": performance,
            }
        )

    except Exception as e:
        print(f"Error in optimization: {str(e)}")  # Debug print
        return jsonify({"success": False, "error": str(e)})


@app.route("/backtesting")
def backtesting():
    return render_template("backtesting.html")


@app.route("/api/backtest", methods=["POST"])
def run_backtest():
    try:
        data = request.json
        config = PortfolioConfig(
            initial_capital=float(data["initial_capital"]),
            lookback_months=int(data["lookback_months"]),
            total_months=int(data["total_months"]),
            start_year=int(data["start_year"]),
            tickers=data["tickers"],
            target_return=(
                float(data["target_return"]) if data["target_return"] else None
            ),
        )

        simulator = PortfolioSimulator(config)
        simulator.run_simulation()

        chart = create_portfolio_chart(simulator.portfolio_values, simulator.dates)

        return jsonify(
            {
                "success": True,
                "summary": simulator.get_summary_statistics(),
                "chart": chart,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
