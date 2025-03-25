import pandas as pd
from scripts.utils import load_config, save_df, logging, days_elapsed

def is_stock_related(text):
    """Check if text is stock-related."""
    keywords = ["stock", "shares", "market", "earnings", "revenue", "profit", "loss", "price"]
    return any(k in str(text).lower() for k in keywords)

def preprocess_news(df, text_cols, time_col, category):
    """Preprocess news data."""
    try:
        # Combine text columns
        df["cleaned_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)
        # Basic cleaning
        df["cleaned_text"] = df["cleaned_text"].str.lower().str.strip()
        # Filter for stock-related (only for stock news)
        if category == "stock":
            df = df[df["cleaned_text"].apply(is_stock_related)]
        # Standardize timestamp to date only
        df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce").dt.date
        df["days_elapsed"] = df["timestamp"].apply(lambda x: days_elapsed(x) if pd.notna(x) else 0)
        # Drop rows with NaN timestamps
        df = df.dropna(subset=["timestamp"])
        logging.info(f"Preprocessed {category} news successfully. Rows: {len(df)}")
        # Log sample timestamps for debugging
        logging.debug(f"Sample {category} timestamps: {df['timestamp'].head().tolist()}")
        return df[["cleaned_text", "timestamp", "days_elapsed"]]
    except Exception as e:
        logging.error(f"Preprocessing failed for {category}: {e}")
        raise

def main():
    config = load_config()
    # Load raw data
    stock_news = pd.read_csv("data/raw/stock_news_raw_20250324.csv")
    country_news = pd.read_csv("data/raw/country_news_raw_20250324.csv")
    market_news = pd.read_csv("data/raw/market_news_raw_20250324.csv")
    # Preprocess
    stock_processed = preprocess_news(stock_news, ["title", "summary"], "time_published", "stock")
    country_processed = preprocess_news(country_news, ["title", "description"], "publishedAt", "country")
    market_processed = preprocess_news(market_news, ["title", "description"], "publishedAt", "market")
    # Save processed data
    save_df(stock_processed, "processed", "stock_processed")
    save_df(country_processed, "processed", "country_processed")
    save_df(market_processed, "processed", "market_processed")

if __name__ == "__main__":
    main()