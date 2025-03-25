import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scripts.utils import load_config, save_df, logging

def fetch_alpha_vantage_news(config):
    """Fetch stock news from Alpha Vantage."""
    url = (f"{config['alpha_vantage']['base_url']}?function=NEWS_SENTIMENT"
           f"&tickers={config['tickers']['stock']}&limit=1000"
           f"&apikey={config['alpha_vantage']['api_key']}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data["feed"])[["title", "summary", "time_published"]]
        logging.info("Fetched Alpha Vantage news successfully.")
        return df
    except Exception as e:
        logging.warning(f"Alpha Vantage API failed: {e}. Using synthetic data.")
        return synthetic_stock_news()

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

def fetch_volatilities(config):
    """Fetch 30-day volatilities from yfinance."""
    tickers = [config["tickers"]["stock"], config["tickers"]["country"], config["tickers"]["market"]]
    try:
        data = yf.download(tickers, period="30d")["Adj Close"]
        volatilities = data.pct_change().std()
        df = pd.DataFrame({"ticker": tickers, "volatility": volatilities.values})
        logging.info("Fetched volatilities successfully.")
        return df
    except Exception as e:
        logging.warning(f"yfinance failed: {e}. Using synthetic volatilities.")
        return pd.DataFrame({
            "ticker": tickers,
            "volatility": [0.02, 0.015, 0.018]
        })

def synthetic_stock_news():
    """Generate synthetic stock news."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq="H")
    titles = [f"AAPL stock {w}" for w in ["surges", "drops", "stable"] * 33 + ["rises"]]
    summaries = [f"Apple stock news: {t}" for t in titles]
    return pd.DataFrame({
        "title": titles,
        "summary": summaries,
        "time_published": dates.strftime("%Y-%m-%dT%H:%M:%SZ")
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
    # Fetch data
    stock_news = fetch_alpha_vantage_news(config)
    country_news = fetch_newsapi_articles(config, config["queries"]["country"])
    market_news = fetch_newsapi_articles(config, config["queries"]["market"])
    volatilities = fetch_volatilities(config)
    # Save raw data
    save_df(stock_news, "raw", "stock_news_raw")
    save_df(country_news, "raw", "country_news_raw")
    save_df(market_news, "raw", "market_news_raw")
    save_df(volatilities, "raw", "volatilities")

if __name__ == "__main__":
    main()