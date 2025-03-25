import pandas as pd
from transformers import pipeline
from scripts.utils import load_config, save_df, logging

SENTIMENT_KEYWORDS = {
    "positive": ["surge", "gain", "profit", "rise", "strong"],
    "negative": ["drop", "loss", "decline", "weak", "crash"]
}

def analyze_sentiment(texts):
    """Analyze sentiment using FinBERT with confidence-based scoring."""
    try:
        classifier = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", truncation=True, max_length=512)
        results = classifier(texts)
        sentiments = []
        confidences = []
        for res, text in zip(results, texts):
            label = res["label"]
            confidence = res["score"]  # This is the model's confidence (0 to 1)
            
            # Use confidence to scale sentiment
            if label == "Positive":
                sentiment = confidence  # Will be between 0 and 1
            elif label == "Negative":
                sentiment = -confidence  # Will be between -1 and 0
            else:  # Neutral
                # For neutral, use keyword analysis but scale by confidence
                pos_count = sum(1 for w in SENTIMENT_KEYWORDS["positive"] if w in text.lower())
                neg_count = sum(1 for w in SENTIMENT_KEYWORDS["negative"] if w in text.lower())
                if pos_count > neg_count:
                    sentiment = confidence * 0.5  # Scaled down for neutral cases
                elif neg_count > pos_count:
                    sentiment = -confidence * 0.5
                else:
                    sentiment = 0
            
            sentiments.append(sentiment)
            confidences.append(confidence)
        
        logging.info("Sentiment analysis completed successfully.")
        return sentiments, confidences
    except Exception as e:
        logging.error(f"Sentiment analysis failed: {e}")
        raise

def main():
    config = load_config()
    # Load processed data
    for category in ["stock", "country", "market"]:
        df = pd.read_csv(f"data/processed/{category}_processed_20250324.csv")
        if df.empty:
            logging.warning(f"No data found for {category}. Skipping sentiment analysis.")
            continue
        sentiments, confidences = analyze_sentiment(df["cleaned_text"].tolist())
        df["sentiment_score"] = sentiments
        df["confidence"] = confidences
        # Ensure no NaN in sentiment_score
        df["sentiment_score"] = df["sentiment_score"].fillna(0)
        save_df(df, "processed", f"{category}_sentiment")

if __name__ == "__main__":
    main()