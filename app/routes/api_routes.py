# routes/api_routes.py
from flask import Blueprint, jsonify, request
from services.market_service import get_correlation_and_clusters
from services.stock_service import get_stock_data
from services.portfolio_service import run_optimization, run_backtest
from services.chart_service import create_portfolio_chart
from datetime import datetime
import logging

api_bp = Blueprint("api", __name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@api_bp.route("/stock-info/<ticker>")
def get_stock_info(ticker):
    try:
        basic_info, hist_data, price_chart, news = get_stock_data(ticker)
        return jsonify(
            {"success": True, "info": basic_info, "chart": price_chart, "news": news}
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@api_bp.route("/optimize-portfolio", methods=["POST"])
def optimize_portfolio():
    try:
        data = request.json
        logger.debug(f"Received request data: {data}")

        # Validate input data
        required_fields = ["tickers", "initial_capital", "start_date", "end_date"]
        for field in required_fields:
            if field not in data:
                return jsonify(
                    {"success": False, "error": f"Missing required field: {field}"}
                )

        if not isinstance(data["tickers"], list) or len(data["tickers"]) == 0:
            return jsonify(
                {"success": False, "error": "Tickers must be a non-empty array"}
            )

        result = run_optimization(
            data["tickers"],
            float(data["initial_capital"]),
            datetime.strptime(data["start_date"], "%Y-%m-%d"),
            datetime.strptime(data["end_date"], "%Y-%m-%d"),
            float(data["target_return"]) if data.get("target_return") else None,
        )

        logger.debug(f"Optimization result: {result}")

        if not result:
            return jsonify(
                {"success": False, "error": "Optimization returned no results"}
            )

        response = {
            "success": True,
            "result": {
                "weights": result.get("weights", {}),
                "performance": result.get("performance", {}),
                "tickers": result.get("tickers", data["tickers"]),
            },
        }

        logger.debug(f"Sending response: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in optimization: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)})


@api_bp.route("/backtest", methods=["POST"])
def backtest():
    try:
        data = request.json
        summary, chart = run_backtest(data)
        return jsonify({"success": True, "summary": summary, "chart": chart})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@api_bp.route("/market-correlation", methods=["POST"])
def get_market_correlation():
    try:
        data = request.json
        logger.debug(f"Received correlation request data: {data}")

        # Validate input data
        if not data.get("tickers") or not isinstance(data["tickers"], list):
            return jsonify({"success": False, "error": "Invalid tickers provided"})

        if not data.get("start_date") or not data.get("end_date"):
            return jsonify(
                {"success": False, "error": "Start and end dates are required"}
            )

        # Import the analysis function here to avoid circular imports
        from services.market_service import get_correlation_and_clusters

        result = get_correlation_and_clusters(
            data["tickers"], data["start_date"], data["end_date"]
        )

        if not result.get("success"):
            return jsonify(result)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in correlation analysis: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)})
