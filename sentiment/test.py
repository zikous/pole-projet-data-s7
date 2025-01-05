import yfinance as yf
from transformers import pipeline
import matplotlib.pyplot as plt

# Load FinBERT sentiment analysis pipeline
finbert_sentiment = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")


# Function to get stock data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    return stock.history(start=start_date, end=end_date)


# Function to fetch news
def get_news(ticker):
    stock = yf.Ticker(ticker)
    return stock.get_news()


# Analyze sentiment using FinBERT
def analyze_sentiment_with_finbert(text):
    result = finbert_sentiment(text)
    return result[0]["label"]


# Main function
def analyze_stock_sentiment(ticker, start_date, end_date):
    # Fetch stock data
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Fetch news articles
    news_articles = get_news(ticker)

    # Perform sentiment analysis on news
    analyzed_news = []
    for article in news_articles:
        title = article.get("title", "")
        content = article.get("content", "")
        full_text = f"{title}. {content}"
        sentiment = analyze_sentiment_with_finbert(full_text)
        analyzed_news.append(
            {
                "title": title,
                "content": content,
                "link": article.get("link", ""),
                "sentiment": sentiment,
            }
        )

    return stock_data, analyzed_news


# Plot stock prices
def plot_stock_prices(stock_data):
    stock_data["Close"].plot(title="Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.show()


# Run the analysis
if __name__ == "__main__":
    # Parameters
    TICKER = "AAPL"  # Replace with the stock ticker you want to analyze
    START_DATE = "2023-12-01"
    END_DATE = "2023-12-31"

    # Get data and sentiment
    stock_data, analyzed_news = analyze_stock_sentiment(TICKER, START_DATE, END_DATE)

    # Show stock data
    print("Stock Data:")
    print(stock_data)

    # Show news and sentiment
    print("\nNews Sentiment Analysis:")
    for news in analyzed_news:
        print(f"Title: {news['title']}")
        print(f"Content: {news['content']}")
        print(f"Link: {news['link']}")
        print(f"Sentiment: {news['sentiment']}")
        print("-" * 80)

    # Plot stock prices
    plot_stock_prices(stock_data)
