import os
from functools import lru_cache
import time
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

# Constants for model, file paths, and settings
MODEL_NAME = (
    "yiyanghkust/finbert-tone"  # Pre-trained FinBERT model for sentiment analysis
)
MODEL_DIR = "./models/finbert-tone"  # Directory to store the model locally
MAX_TEXT_LENGTH = 500  # Max length of text for sentiment analysis
REQUEST_TIMEOUT = 10  # Timeout for HTTP requests
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Python-requests/2.28.1)"}

# Initialize NLTK once to avoid repeated downloads
nltk.download("stopwords", quiet=True)  # Download stopwords for text processing
stop_words = set(stopwords.words("english"))  # Set of English stopwords

# Compile regular expressions for text cleaning
SPECIAL_CHARS_PATTERN = re.compile(
    r"[^\w\s]"
)  # Matches special characters (non-word and non-space)
NUMBERS_PATTERN = re.compile(r"\d+")  # Matches numbers


# Function to load or download the FinBERT model with caching
@lru_cache(maxsize=1)  # Cache the model loading to avoid reloading for subsequent calls
def load_or_download_model():
    """Try to load local model first, if fails download it again."""
    try:
        # Try loading from local directory first
        if os.path.exists(MODEL_DIR):
            try:
                return pipeline("sentiment-analysis", model=MODEL_DIR)
            except Exception as e:
                print(f"Error loading local model: {str(e)}")
                print("Attempting to redownload model...")

        # If local load fails or directory doesn't exist, download and save
        os.makedirs(MODEL_DIR, exist_ok=True)
        model = pipeline("sentiment-analysis", model=MODEL_NAME)
        model.save_pretrained(MODEL_DIR)
        return model

    except Exception as e:
        print(f"Error in load_or_download_model: {str(e)}")
        raise


# Try loading the model at module level to ensure it's available for use
try:
    finbert_sentiment = load_or_download_model()  # Initialize the sentiment model
except Exception as e:
    print(
        f"Failed to load sentiment model: {str(e)}"
    )  # Log failure if the model fails to load
    raise


def clean_text(text: str) -> str:
    """Optimized text cleaning function."""

    # Check if the text is empty or None, return an empty string if true
    if not text:
        return ""

    # Apply regex substitutions to clean the text:
    # - Remove special characters and convert text to lowercase
    text = SPECIAL_CHARS_PATTERN.sub(" ", text.lower())
    # - Remove numbers from the text
    text = NUMBERS_PATTERN.sub(" ", text)

    # Efficient word filtering:
    # - Split the text into words
    # - Retain only words that are not in stopwords, have more than 2 characters, and are non-empty
    words = [
        word
        for word in text.split()
        if word and word not in stop_words and len(word) > 2
    ]

    # Join the filtered words back into a single string with spaces in between
    return " ".join(words)


def truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Efficiently truncate text to maximum length."""

    # Check if the text length is already within the maximum limit
    if len(text) <= max_length:
        return (
            text  # No truncation needed if text is shorter than or equal to max_length
        )

    # If text is too long, truncate it at the last complete word within the max_length
    return text[:max_length].rsplit(" ", 1)[0]


def fetch_article_content(url: str) -> Optional[str]:
    """Fetch article content with timeout and error handling."""

    try:
        # Send a GET request to the URL with a timeout and custom headers
        with requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS) as response:

            # Check if the HTTP status code is 200 (OK)
            if response.status_code != 200:
                return None  # If not 200, return None to indicate failure

            # Use SoupStrainer to limit parsing to only <p> tags
            soup = BeautifulSoup(
                response.content,
                "html.parser",
                parse_only=SoupStrainer("p"),  # Parse only <p> tags for content
            )

            # Extract and join text from all <p> tags, removing any empty or whitespace-only text
            return " ".join(
                p.get_text() for p in soup.find_all("p") if p.get_text().strip()
            )

    except Exception as e:
        # Log the error in case of failure (e.g., network issues, invalid response)
        print(f"Error fetching article content: {str(e)}")
        return None  # Return None to indicate an error


def process_news_item(item: Dict) -> Optional[Dict]:
    """Process a single news item with sentiment analysis."""

    try:
        # Check if the input item is a dictionary and contains 'title' and 'link'
        if not isinstance(item, dict) or "title" not in item or "link" not in item:
            return None  # Return None if the item is not in the expected format

        # Extract the title and content from the news item
        title = item["title"]
        content = fetch_article_content(
            item["link"]
        )  # Fetch article content from the URL

        # Combine title and content for sentiment analysis, if content is available
        sentiment_text = f"{title} {content}" if content else title

        # Clean the combined text to remove unwanted characters and words
        cleaned_text = clean_text(sentiment_text)

        # Truncate the cleaned text to a maximum length
        truncated_text = truncate_text(cleaned_text)

        # If the truncated text is empty, return None as no meaningful text exists
        if not truncated_text:
            return None

        # Perform sentiment analysis using the FinBERT model
        sentiment = finbert_sentiment(truncated_text)[0]

        # Return the processed result, including the sentiment label and score
        return {
            "title": title,
            "link": item["link"],
            "sentiment": sentiment["label"],
            "score": sentiment["score"],
        }

    except Exception as e:
        # Handle any errors that occur during the processing of the news item
        print(f"Error processing news item: {str(e)}")
        return None  # Return None if an error occurs


def get_stock_data(ticker: str) -> Tuple[Dict, object, object, List[Dict]]:
    """Optimized stock data retrieval function."""

    try:
        # Fetch stock data using yfinance Ticker object
        stock = yf.Ticker(ticker)

        # Retrieve basic information and historical data concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit the fetching tasks for stock info and historical data
            info_future = executor.submit(lambda: stock.info)
            hist_future = executor.submit(lambda: stock.history(period="1y"))

            # Wait for results to be fetched
            info = info_future.result()
            hist = hist_future.result()

        # Process the basic information and extract relevant details
        basic_info = {
            "name": info.get("longName", "N/A"),  # Full company name
            "sector": info.get("sector", "N/A"),  # Sector the company belongs to
            "market_cap": info.get("marketCap", "N/A"),  # Market capitalization
            "pe_ratio": info.get("trailingPE", "N/A"),  # Price-to-earnings ratio
            "dividend_yield": info.get("dividendYield", "N/A"),  # Dividend yield
            "beta": info.get("beta", "N/A"),  # Stock's beta value (volatility measure)
            "profit_margin": info.get("profitMargins", "N/A"),  # Profit margin
            "debt_to_equity": info.get("debtToEquity", "N/A"),  # Debt-to-equity ratio
            "roe": info.get("returnOnEquity", "N/A"),  # Return on equity
            "roa": info.get("returnOnAssets", "N/A"),  # Return on assets
        }

        # If historical data exists, calculate technical indicators (50-day and 200-day SMAs)
        if not hist.empty:
            hist["SMA_50"] = (
                hist["Close"].rolling(window=50).mean()
            )  # 50-day Simple Moving Average
            hist["SMA_200"] = (
                hist["Close"].rolling(window=200).mean()
            )  # 200-day Simple Moving Average

        # Attempt to create a price chart with historical data
        try:
            price_chart = create_price_chart(
                hist
            )  # Create a chart for stock price over time
        except Exception as e:
            print(
                f"Error creating price chart: {str(e)}"
            )  # Catch any errors during chart creation
            price_chart = None  # Set price chart to None if there is an error

        # Process the latest news articles concurrently
        news = []
        try:
            # Limit the number of news articles
            stock_news = stock.news[:]

            # Process each news item sequentially
            news = [
                process_news_item(item)
                for item in stock_news
                if process_news_item(item) is not None
            ]

        except Exception as e:
            print(
                f"Error processing news: {str(e)}"
            )  # Catch errors related to news processing

        # Return the processed data: basic info, historical data, price chart, and news
        return basic_info, hist, price_chart, news

    except Exception as e:
        print(
            f"Error fetching stock data for {ticker}: {str(e)}"
        )  # Handle any exceptions during stock data retrieval
        raise  # Re-raise the exception for upstream handling
