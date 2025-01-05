import os
from transformers import pipeline
import yfinance as yf
from services.chart_service import create_price_chart

# Define the model path
MODEL_NAME = "yiyanghkust/finbert-tone"
MODEL_DIR = "./models/finbert-tone"


def load_or_download_model():
    """Load FinBERT model from local storage or download if not available."""
    if not os.path.exists(MODEL_DIR):
        print("Model not found locally. Downloading...")
        # Download and save the model locally
        pipeline("sentiment-analysis", model=MODEL_NAME).save_pretrained(MODEL_DIR)
        print("Model downloaded and saved locally.")
    else:
        print("Model found locally. Loading...")
    # Load the model from the local directory
    return pipeline("sentiment-analysis", model=MODEL_DIR)


# Load the FinBERT sentiment analysis model
finbert_sentiment = load_or_download_model()


def get_stock_data(ticker):
    """Get stock information, historical data, and news with sentiment analysis."""
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
        for item in stock_news:
            title = item["title"]
            sentiment = finbert_sentiment(title)[0]  # Analyze sentiment of the title
            news.append(
                {
                    "title": title,
                    "link": item["link"],
                    "sentiment": sentiment[
                        "label"
                    ],  # "LABEL_0" -> Negative, "LABEL_1" -> Neutral, "LABEL_2" -> Positive
                    "score": sentiment["score"],  # Confidence score
                }
            )
    except Exception as e:
        print(f"Error fetching news: {e}")

    return basic_info, hist, price_chart, news
