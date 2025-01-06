import os
from functools import lru_cache
from transformers import pipeline
import yfinance as yf
from services.chart_service import create_price_chart
import re
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional

# Constants
MODEL_NAME = "yiyanghkust/finbert-tone"
MODEL_DIR = "./models/finbert-tone"
MAX_TEXT_LENGTH = 500
MAX_NEWS_ITEMS = 10
REQUEST_TIMEOUT = 10
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Initialize NLTK once
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# Compile regex patterns once
SPECIAL_CHARS_PATTERN = re.compile(r"[^\w\s]")
NUMBERS_PATTERN = re.compile(r"\d+")


@lru_cache(maxsize=1)
def load_or_download_model():
    """Cache the model loading to avoid reloading for subsequent calls."""
    try:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR, exist_ok=True)
            pipeline("sentiment-analysis", model=MODEL_NAME).save_pretrained(MODEL_DIR)
        return pipeline("sentiment-analysis", model=MODEL_DIR)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


# Load model once at module level
try:
    finbert_sentiment = load_or_download_model()
except Exception as e:
    print(f"Failed to load sentiment model: {str(e)}")
    raise


def clean_text(text: str) -> str:
    """Optimized text cleaning function."""
    if not text:
        return ""

    # Apply regex substitutions
    text = SPECIAL_CHARS_PATTERN.sub(" ", text.lower())
    text = NUMBERS_PATTERN.sub(" ", text)

    # Efficient word filtering
    words = [
        word
        for word in text.split()
        if word and word not in stop_words and len(word) > 2
    ]

    return " ".join(words)


def truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Efficiently truncate text to maximum length."""
    if len(text) <= max_length:
        return text

    return text[:max_length].rsplit(" ", 1)[0]


def fetch_article_content(url: str) -> Optional[str]:
    """Fetch article content with timeout and error handling."""
    try:
        with requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS) as response:
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(
                response.content,
                "html.parser",
                parse_only=BeautifulSoup.SoupStrainer("p"),
            )
            return " ".join(
                p.get_text() for p in soup.find_all("p") if p.get_text().strip()
            )
    except Exception as e:
        print(f"Error fetching article content: {str(e)}")
        return None


def process_news_item(item: Dict) -> Optional[Dict]:
    """Process a single news item with sentiment analysis."""
    try:
        if not isinstance(item, dict) or "title" not in item or "link" not in item:
            return None

        title = item["title"]
        content = fetch_article_content(item["link"])
        sentiment_text = f"{title} {content}" if content else title

        cleaned_text = clean_text(sentiment_text)
        truncated_text = truncate_text(cleaned_text)

        if not truncated_text:
            return None

        sentiment = finbert_sentiment(truncated_text)[0]

        return {
            "title": title,
            "link": item["link"],
            "sentiment": sentiment["label"],
            "score": sentiment["score"],
        }
    except Exception as e:
        print(f"Error processing news item: {str(e)}")
        return None


def get_stock_data(ticker: str) -> Tuple[Dict, object, object, List[Dict]]:
    """Optimized stock data retrieval function."""
    try:
        stock = yf.Ticker(ticker)

        # Get basic info and history concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            info_future = executor.submit(lambda: stock.info)
            hist_future = executor.submit(lambda: stock.history(period="1y"))

            info = info_future.result()
            hist = hist_future.result()

        # Process basic info
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

        # Calculate technical indicators if data exists
        if not hist.empty:
            hist["SMA_50"] = hist["Close"].rolling(window=50).mean()
            hist["SMA_200"] = hist["Close"].rolling(window=200).mean()

        # Create price chart
        try:
            price_chart = create_price_chart(hist)
        except Exception as e:
            print(f"Error creating price chart: {str(e)}")
            price_chart = None

        # Process news items concurrently
        news = []
        try:
            stock_news = stock.news[:MAX_NEWS_ITEMS]
            with ThreadPoolExecutor(max_workers=min(len(stock_news), 5)) as executor:
                news_futures = [
                    executor.submit(process_news_item, item) for item in stock_news
                ]

                news = [
                    result.result()
                    for result in news_futures
                    if result.result() is not None
                ]
        except Exception as e:
            print(f"Error processing news: {str(e)}")

        return basic_info, hist, price_chart, news

    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {str(e)}")
        raise
