import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scripts.utils import load_config, save_df, logging
import time

def fetch_alpha_vantage_news(config, ticker):
    """Fetch stock news from Alpha Vantage for a specific ticker with retry and rate limiting."""
    url = (f"{config['alpha_vantage']['base_url']}?function=NEWS_SENTIMENT"
           f"&tickers={ticker}&limit=1000"
           f"&apikey={config['alpha_vantage']['api_key']}")
    for attempt in range(3):  # Retry up to 3 times
        try:
            time.sleep(12)  # 12-second delay to respect 5 calls/minute limit
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            logging.debug(f"Alpha Vantage response for {ticker}: {data}")  # Log raw response for debugging
            if "feed" not in data or not data["feed"]:
                raise ValueError("Empty or invalid feed response.")
            df = pd.DataFrame(data["feed"])
            required_cols = ["title", "summary", "time_published"]
            if not all(col in df.columns for col in required_cols):
                raise KeyError(f"Missing required columns: {required_cols}")
            df = df[required_cols]
            df["ticker"] = ticker
            logging.info(f"Fetched Alpha Vantage news for {ticker} successfully.")
            return df
        except Exception as e:
            logging.warning(f"Alpha Vantage API failed for {ticker} (attempt {attempt + 1}): {e}")
            if attempt == 2:  # Last attempt
                logging.warning(f"Using synthetic data for {ticker} after retries.")
                return synthetic_stock_news(ticker)
    return synthetic_stock_news(ticker)  # Fallback if all retries fail

def fetch_newsapi_articles(config, query):
    """Fetch news from NewsAPI."""
    url = (f"{config['newsapi']['base_url']}?q={query}"
           f"&from={(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')}"
           f"&apiKey={config['newsapi']['api_key']}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data["articles"])[["title", "description", "publishedAt"]]
        logging.info(f"Fetched NewsAPI articles for query '{query}' successfully.")
        return df
    except Exception as e:
        logging.warning(f"NewsAPI failed for query '{query}': {e}. Using synthetic data.")
        return synthetic_news(query)

def synthetic_stock_news(ticker):
    """Generate synthetic stock news."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq="H")
    titles = [f"{ticker} stock {w}" for w in ["surges", "drops", "stable"] * 33 + ["rises"]]
    summaries = [f"{ticker} stock news: {t}" for t in titles]
    return pd.DataFrame({
        "title": titles,
        "summary": summaries,
        "time_published": dates.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ticker": ticker
    })

def synthetic_news(query):
    """Generate synthetic news for country/market."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq="H")
    titles = [f"{query} {w}" for w in ["grows", "declines", "steady"] * 33 + ["improves"]]
    descriptions = [f"{query} update: {t}" for t in titles]
    return pd.DataFrame({
        "title": titles,
        "description": descriptions,
        "publishedAt": dates.strftime("%Y-%m-%dT%H:%M:%SZ")
    })

def main():
    config = load_config()
    stock_dfs = [fetch_alpha_vantage_news(config, ticker) for ticker in config["tickers"]["stocks"]]
    stock_news = pd.concat(stock_dfs, ignore_index=True)
    country_news = fetch_newsapi_articles(config, config["queries"]["country"])
    market_news = fetch_newsapi_articles(config, config["queries"]["market"])
    
    save_df(stock_news, "raw", "stock_news_raw")
    save_df(country_news, "raw", "country_news_raw")
    save_df(market_news, "raw", "market_news_raw")

if __name__ == "__main__":
    main()