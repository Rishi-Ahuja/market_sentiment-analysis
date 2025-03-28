import pandas as pd
from scripts.utils import load_config, save_df, logging, days_elapsed

def is_stock_related(text):
    """Check if text is stock-related."""
    keywords = ["stock", "shares", "market", "earnings", "revenue", "profit", "loss", "price"]
    return any(k in str(text).lower() for k in keywords)

def preprocess_news(df, text_cols, time_col, category, ticker=None):
    """Preprocess news data."""
    try:
        df["cleaned_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)
        df["cleaned_text"] = df["cleaned_text"].str.lower().str.strip()
        if category == "stock" and ticker:
            df = df[df["cleaned_text"].apply(is_stock_related) & df["ticker"].eq(ticker)]
        df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
        df["days_elapsed"] = df["timestamp"].apply(days_elapsed)
        df = df.dropna(subset=["timestamp"])
        logging.info(f"Preprocessed {category} news for {ticker or 'all'}: {len(df)} rows")
        return df[["cleaned_text", "timestamp", "days_elapsed", "ticker"] if category == "stock" else ["cleaned_text", "timestamp", "days_elapsed"]]
    except Exception as e:
        logging.error(f"Preprocessing failed for {category} ({ticker or 'all'}): {e}")
        raise

def main():
    config = load_config()
    stock_news = pd.read_csv("data/raw/stock_news_raw.csv")
    country_news = pd.read_csv("data/raw/country_news_raw.csv")
    market_news = pd.read_csv("data/raw/market_news_raw.csv")
    
    for ticker in config["tickers"]["stocks"]:
        stock_processed = preprocess_news(stock_news, ["title", "summary"], "time_published", "stock", ticker)
        save_df(stock_processed, "processed", "stock_processed", ticker)
    country_processed = preprocess_news(country_news, ["title", "description"], "publishedAt", "country")
    market_processed = preprocess_news(market_news, ["title", "description"], "publishedAt", "market")
    
    save_df(country_processed, "processed", "country_processed")
    save_df(market_processed, "processed", "market_processed")

if __name__ == "__main__":
    main()