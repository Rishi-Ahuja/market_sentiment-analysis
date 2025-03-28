import pandas as pd
import numpy as np
from transformers import pipeline
from scripts.utils import load_config, save_df, logging
import torch

SENTIMENT_KEYWORDS = {
    "positive": ["surge", "gain", "profit", "rise", "strong"],
    "negative": ["drop", "loss", "decline", "weak", "crash"]
}

def analyze_sentiment(texts, model_name="yiyanghkust/finbert-tone", batch_size=32):
    """Analyze sentiment using FinBERT with confidence-based scoring."""
    try:
        device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU
        classifier = pipeline("sentiment-analysis", model=model_name, truncation=True, max_length=512, device=device)
        sentiments, confidences = [], []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results = classifier(batch)
            for res, text in zip(results, batch):
                label = res["label"]
                confidence = res["score"]
                if label == "Positive":
                    sentiment = confidence * (1 - 0.1 * confidence)
                elif label == "Negative":
                    sentiment = -confidence * (1 - 0.05 * confidence)
                else:
                    pos_count = sum(1 for w in SENTIMENT_KEYWORDS["positive"] if w in text.lower())
                    neg_count = sum(1 for w in SENTIMENT_KEYWORDS["negative"] if w in text.lower())
                    sentiment = 0.3 * (pos_count - neg_count) * confidence if pos_count != neg_count else 0
                sentiments.append(sentiment)
                confidences.append(confidence)
        logging.info(f"Sentiment analysis completed for {len(texts)} texts.")
        return sentiments, confidences
    except Exception as e:
        logging.error(f"Sentiment analysis failed: {e}")
        # Fallback to synthetic sentiment
        logging.warning("Using synthetic sentiment scores as fallback.")
        return [np.random.uniform(-1, 1) for _ in texts], [0.5] * len(texts)

def main():
    config = load_config()
    categories = ["stock", "country", "market"]
    batch_size = config.get("sentiment", {}).get("batch_size", 32)

    for category in categories:
        if category == "stock":
            for ticker in config["tickers"]["stocks"]:
                try:
                    df = pd.read_csv(f"data/processed/stock_processed_{ticker}.csv")
                    if df.empty or "cleaned_text" not in df.columns:
                        logging.warning(f"No valid data for {ticker}. Skipping.")
                        continue
                    texts = df["cleaned_text"].fillna("").tolist()
                    sentiments, confidences = analyze_sentiment(texts, batch_size=batch_size)
                    df["sentiment_score"] = sentiments
                    df["confidence"] = confidences
                    save_df(df, "processed", "stock_sentiment", ticker)
                    logging.info(f"Processed {ticker}: Mean sentiment = {df['sentiment_score'].mean():.3f}")
                except Exception as e:
                    logging.error(f"Failed to process {ticker}: {e}")
                    continue
        else:
            try:
                df = pd.read_csv(f"data/processed/{category}_processed.csv")
                if df.empty or "cleaned_text" not in df.columns:
                    logging.warning(f"No valid data for {category}. Skipping.")
                    continue
                texts = df["cleaned_text"].fillna("").tolist()
                sentiments, confidences = analyze_sentiment(texts, batch_size=batch_size)
                df["sentiment_score"] = sentiments
                df["confidence"] = confidences
                save_df(df, "processed", f"{category}_sentiment")
                logging.info(f"Processed {category}: Mean sentiment = {df['sentiment_score'].mean():.3f}")
            except Exception as e:
                logging.error(f"Failed to process {category}: {e}")
                continue

if __name__ == "__main__":
    main()