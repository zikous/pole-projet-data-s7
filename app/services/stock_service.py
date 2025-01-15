import os
from functools import lru_cache
from transformers import pipeline
import yfinance as yf
from services.chart_service import create_price_chart
import re
from bs4 import BeautifulSoup, SoupStrainer
import requests
import nltk
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional

# Constants
USE_GPU = True  # Set to False if you want to use CPU
MODEL_NAME = "yiyanghkust/finbert-tone"
MODEL_DIR = "./models/finbert-tone"
MAX_TEXT_LENGTH = 500
REQUEST_TIMEOUT = 10
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Python-requests/2.28.1)"}

# Initialize NLTK
try:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
except Exception as e:
    print(f"Error downloading stopwords: {str(e)}")
    stop_words = set()

# Compile regex patterns
SPECIAL_CHARS_PATTERN = re.compile(r"[^\w\s]")
NUMBERS_PATTERN = re.compile(r"\d+")


def clean_text(text: str) -> str:
    """Clean text removing special characters and stopwords."""
    if not text:
        return ""

    try:
        text = SPECIAL_CHARS_PATTERN.sub(" ", text.lower())
        text = NUMBERS_PATTERN.sub(" ", text)
        words = [
            word
            for word in text.split()
            if word and word not in stop_words and len(word) > 2
        ]
        return " ".join(words)
    except Exception as e:
        print(f"Error cleaning text: {str(e)}")
        return ""


def truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Truncate text to maximum length at word boundary."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0]


def load_or_download_model():
    """Try to load local model first, if fails download it again."""
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Set device according to USE_GPU flag
        device = 0 if USE_GPU else -1  # 0 for GPU, -1 for CPU

        if os.path.exists(MODEL_DIR) and os.path.isfile(
            os.path.join(MODEL_DIR, "config.json")
        ):
            try:
                return pipeline("sentiment-analysis", model=MODEL_DIR, device=device)
            except Exception as e:
                print(f"Error loading local model: {str(e)}")
                print("Attempting to redownload model...")

        model = pipeline("sentiment-analysis", model=MODEL_NAME, device=device)
        model.save_pretrained(MODEL_DIR)
        return model

    except Exception as e:
        print(f"Error in load_or_download_model: {str(e)}")
        raise


# Global model instance
finbert_sentiment = None
try:
    finbert_sentiment = load_or_download_model()
except Exception as e:
    print(f"Failed to load sentiment model: {str(e)}")


def fetch_article_content(url: str) -> Optional[str]:
    """Fetch article content with timeout and error handling."""
    try:
        with requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS) as response:
            if response.status_code != 200:
                return None

            # Try to parse the full article text from <p> tags
            soup = BeautifulSoup(
                response.content, "html.parser", parse_only=SoupStrainer("p")
            )
            paragraphs = [
                p.get_text() for p in soup.find_all("p") if p.get_text().strip()
            ]

            if paragraphs:
                return " ".join(paragraphs)
            else:
                return None

    except Exception as e:
        print(f"Error fetching article content: {str(e)}")
        return None


def process_news_item(item: Dict) -> Optional[Dict]:
    """Process a single news item with sentiment analysis."""
    global finbert_sentiment

    try:
        if (
            not isinstance(item, dict)
            or "content" not in item
            or "title" not in item["content"]
            or "canonicalUrl" not in item["content"]
        ):
            print(f"Invalid News Item Format: {item}")
            return None

        title = item["content"]["title"]
        link = item["content"]["canonicalUrl"]["url"]

        # Try to fetch the article content using the updated method
        content = fetch_article_content(link)
        print(f"Fetched Content: {content}")

        # If content is fetched successfully, combine it with the title, otherwise use title only
        sentiment_text = f"{title} {content}" if content else title
        cleaned_text = clean_text(sentiment_text)
        print(f"Cleaned Text: {cleaned_text}")

        truncated_text = truncate_text(cleaned_text)
        print(f"Truncated Text: {truncated_text}")

        if not truncated_text:
            print("No text after truncation, skipping sentiment analysis.")
            return None

        # Reload model if it's None
        if finbert_sentiment is None:
            print("Model is not loaded!")
            finbert_sentiment = load_or_download_model()

        sentiment = finbert_sentiment(truncated_text)
        print(f"Sentiment Analysis Result: {sentiment}")

        return {
            "title": title,
            "link": link,
            "sentiment": sentiment[0]["label"],
            "score": float(sentiment[0]["score"]),
        }

    except Exception as e:
        print(f"Error processing news item: {str(e)}")
        return None


def get_stock_data(ticker: str) -> Tuple[Dict, object, object, List[Dict]]:
    """Get stock data with improved error handling."""
    try:
        stock = yf.Ticker(ticker)

        # Fetch stock info and historical data concurrently
        with ThreadPoolExecutor() as executor:
            info_future = executor.submit(lambda: stock.info)
            hist_future = executor.submit(lambda: stock.history(period="1y"))

            info = info_future.result()
            hist = hist_future.result()

        # Process stock information
        basic_info = {
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "market_cap": (
                float(info.get("marketCap", 0)) if info.get("marketCap") else "N/A"
            ),
            "pe_ratio": (
                float(info.get("trailingPE", 0)) if info.get("trailingPE") else "N/A"
            ),
            "dividend_yield": (
                float(info.get("dividendYield", 0))
                if info.get("dividendYield")
                else "N/A"
            ),
            "beta": float(info.get("beta", 0)) if info.get("beta") else "N/A",
            "profit_margin": (
                float(info.get("profitMargins", 0))
                if info.get("profitMargins")
                else "N/A"
            ),
            "debt_to_equity": (
                float(info.get("debtToEquity", 0))
                if info.get("debtToEquity")
                else "N/A"
            ),
            "roe": (
                float(info.get("returnOnEquity", 0))
                if info.get("returnOnEquity")
                else "N/A"
            ),
            "roa": (
                float(info.get("returnOnAssets", 0))
                if info.get("returnOnAssets")
                else "N/A"
            ),
        }

        if not hist.empty:
            hist["SMA_50"] = hist["Close"].rolling(window=50).mean()
            hist["SMA_200"] = hist["Close"].rolling(window=200).mean()

        price_chart = create_price_chart(hist)

        # Process stock news
        stock_news = getattr(stock, "news", []) or []
        news = [
            process_news_item(item) for item in stock_news if process_news_item(item)
        ]

        print(f"Successfully processed {len(news)} news items")
        return basic_info, hist, price_chart, news

    except Exception as e:
        print(f"Error in get_stock_data: {str(e)}")
        raise
