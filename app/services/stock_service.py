# services/stock_service.py
import yfinance as yf
from services.chart_service import create_price_chart


def get_stock_data(ticker):
    """Get stock information, historical data, and news"""
    stock = yf.Ticker(ticker)
    info = stock.info

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

    hist = stock.history(period="1y")
    hist["SMA_50"] = hist["Close"].rolling(window=50).mean()
    hist["SMA_200"] = hist["Close"].rolling(window=200).mean()

    price_chart = create_price_chart(hist)

    news = []
    try:
        stock_news = stock.news[:5]
        news = [{"title": item["title"], "link": item["link"]} for item in stock_news]
    except Exception:
        pass

    return basic_info, hist, price_chart, news
