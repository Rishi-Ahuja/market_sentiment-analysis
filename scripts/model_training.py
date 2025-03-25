import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy.stats import multivariate_normal, skew
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
from scripts.utils import load_config, save_df, exponential_decay, market_impact, logging

class HSAM:
    def __init__(self, decay_rate, pca_components, bayesian_prior_strength, validation_split=0.2):
        self.decay_rate = decay_rate
        self.pca = PCA(n_components=pca_components)
        self.prior_strength = bayesian_prior_strength
        self.weights = None
        self.validation_split = validation_split

    def fit(self, sentiment_data, volatilities):
        """Fit the HSAM model with cross-validation."""
        try:
            # Split data into training and validation sets
            train_size = int(len(sentiment_data) * (1 - self.validation_split))
            train_data = sentiment_data.iloc[:train_size].copy()
            val_data = sentiment_data.iloc[train_size:].copy()
            logging.info(f"Training data: {len(train_data)} rows, Validation data: {len(val_data)} rows")

            # Fit on training data
            train_result = self._fit_subset(train_data, volatilities)
            # Apply to validation data
            val_result = self._fit_subset(val_data, volatilities)

            # Combine results
            result = pd.concat([train_result, val_result], ignore_index=True)
            logging.info("HSAM model fitted successfully.")
            return result
        except Exception as e:
            logging.error(f"HSAM fitting failed: {e}")
            raise

    def _fit_subset(self, data, volatilities):
        """Helper method to fit HSAM on a subset of data."""
        # Ensure no NaN in days_elapsed
        data["days_elapsed"] = data["days_elapsed"].fillna(0)
        sentiment_cols = ["stock", "country", "market"]
        # Check for sufficient data overlap
        valid_rows = data[sentiment_cols].notna().sum(axis=1) >= 2
        if not valid_rows.any():
            logging.error("No rows with sufficient sentiment data (at least 2 sources).")
            raise ValueError("Insufficient overlapping sentiment data.")
        data = data[valid_rows].copy()
        logging.info(f"Rows after overlap filter: {len(data)}")

        # Apply exponential decay to sentiment scores
        data[sentiment_cols] = data.apply(
            lambda row: [exponential_decay(row[col], self.decay_rate, row["days_elapsed"])
                        for col in sentiment_cols], axis=1, result_type="expand"
        )
        # Impute NaN values with 0 before PCA
        imputer = SimpleImputer(strategy="constant", fill_value=0)
        sentiment_matrix = imputer.fit_transform(data[sentiment_cols])
        logging.debug(f"Sentiment matrix shape: {sentiment_matrix.shape}")
        if sentiment_matrix.shape[0] < self.pca.n_components:
            logging.error(f"Not enough samples ({sentiment_matrix.shape[0]}) for PCA with {self.pca.n_components} components.")
            raise ValueError("Insufficient data for PCA.")
        # PCA transformation
        pca_features = self.pca.fit_transform(sentiment_matrix)
        s_pca = pd.DataFrame(pca_features, columns=["P1", "P2"])
        logging.debug(f"s_pca shape: {s_pca.shape}")
        # Initial weights from volatilities
        self.weights = volatilities["volatility"] / volatilities["volatility"].sum()
        self.weights = self.weights[:2].values
        logging.debug(f"Initial weights shape: {self.weights.shape}")
        # Bayesian updating
        weights_dynamic = []
        for i, row in s_pca.iterrows():
            row_values = row.values
            likelihood = multivariate_normal.pdf(row_values, mean=[0, 0], cov=np.eye(2))
            prior = multivariate_normal.pdf(self.weights, mean=[0.5, 0.5], cov=np.eye(2) * self.prior_strength)
            evidence = likelihood * prior
            self.weights = (self.weights * prior + row_values * likelihood) / evidence if evidence != 0 else self.weights
            weights_dynamic.append(self.weights.copy())
        
        # Calculate weighted sentiment scores
        weighted_scores = [np.dot(row, w) for row, w in zip(s_pca.values, weights_dynamic)]
        # Normalize to [-1, 1] range using min-max scaling
        min_score = min(weighted_scores)
        max_score = max(weighted_scores)
        if min_score != max_score:
            normalized_scores = [2 * (score - min_score) / (max_score - min_score) - 1 for score in weighted_scores]
        else:
            normalized_scores = [0] * len(weighted_scores)
        
        # Apply market impact with smaller slope for smoother distribution
        final_sentiment = [market_impact(score, slope=0.3) for score in normalized_scores]
        
        logging.info(f"Final weights: {self.weights}")
        return pd.DataFrame({
            "timestamp": data["timestamp"],
            "stock": data["stock"],
            "country": data["country"],
            "market": data["market"],
            "final_sentiment": final_sentiment
        })

def compute_confidence_intervals(df, n_bootstrap=1000, ci_level=0.95):
    """Compute confidence intervals for daily sentiment scores using bootstrapping on raw data."""
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    ci_lower = []
    ci_upper = []
    daily_groups = df.groupby("date")
    for date, group in daily_groups:
        scores = group["final_sentiment"].values
        bootstrapped_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrapped_means.append(np.mean(sample))
        ci = np.percentile(bootstrapped_means, [(1 - ci_level) / 2 * 100, (1 + ci_level) / 2 * 100])
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])
    daily_sentiment = daily_groups["final_sentiment"].mean().reset_index()
    daily_sentiment["timestamp"] = pd.to_datetime(daily_sentiment["date"])
    daily_sentiment = daily_sentiment.sort_values("timestamp")
    return daily_sentiment, ci_lower, ci_upper

def calibrate_thresholds(final_sentiment, percentile_low=25, percentile_high=75):
    """Calibrate sentiment thresholds based on percentile distribution with fallback."""
    low_threshold = np.percentile(final_sentiment, percentile_low)
    high_threshold = np.percentile(final_sentiment, percentile_high)
    # Fallback if thresholds are at extremes
    if low_threshold <= -1.0:
        low_threshold = -0.3
    if high_threshold >= 1.0:
        high_threshold = 0.3
    logging.info(f"Calibrated thresholds: Negative < {low_threshold:.2f}, Positive > {high_threshold:.2f}")
    return low_threshold, high_threshold

def plot_sentiment(df, smoothing_window=3):
    """Plot sentiment over time with additional visualizations."""
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    raw_sentiments = df[["timestamp", "stock", "country", "market"]].copy()

    # 1. Final Sentiment Plot with Confidence Intervals
    daily_sentiment, ci_lower, ci_upper = compute_confidence_intervals(df)
    smoothed_sentiment = daily_sentiment["final_sentiment"].rolling(window=smoothing_window, min_periods=1).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(daily_sentiment["timestamp"], daily_sentiment["final_sentiment"], label="Daily Sentiment", alpha=0.5, color="blue")
    plt.plot(daily_sentiment["timestamp"], smoothed_sentiment, label=f"Smoothed Sentiment ({smoothing_window}-day)", linewidth=2, color="red")
    plt.fill_between(daily_sentiment["timestamp"], ci_lower, ci_upper, color="blue", alpha=0.1, label="95% CI")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Over Time for AAPL, U.S. Economy, and NASDAQ")
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("data/outputs/final_sentiment_plot.png")
    plt.close()

    # 2. Individual Sentiment Trends (AAPL, U.S. Economy, NASDAQ)
    plt.figure(figsize=(12, 6))
    for col, label in zip(["stock", "country", "market"], ["AAPL", "U.S. Economy", "NASDAQ"]):
        daily = raw_sentiments.groupby(raw_sentiments["timestamp"].dt.date)[col].mean().reset_index()
        daily["timestamp"] = pd.to_datetime(daily["timestamp"])
        daily = daily.sort_values("timestamp")
        plt.plot(daily["timestamp"], daily[col], label=label)
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.title("Individual Sentiment Trends")
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("data/outputs/individual_sentiment_plot.png")
    plt.close()

    # 3. Sentiment Distribution Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df["final_sentiment"], bins=50, color="skyblue", edgecolor="black")
    plt.axvline(df["final_sentiment"].mean(), color="red", linestyle="--", label=f"Mean: {df['final_sentiment'].mean():.2f}")
    plt.axvline(df["final_sentiment"].median(), color="green", linestyle="--", label=f"Median: {df['final_sentiment'].median():.2f}")
    plt.xlabel("Final Sentiment Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Final Sentiment Scores")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/outputs/sentiment_distribution_histogram.png")
    plt.close()

    # 4. Cross-Source Sentiment Correlation Heatmap
    plt.figure(figsize=(8, 6))
    correlation_matrix = raw_sentiments[["stock", "country", "market"]].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title("Cross-Source Sentiment Correlation")
    plt.xticks(ticks=[0, 1, 2], labels=["AAPL", "U.S. Economy", "NASDAQ"], rotation=45)
    plt.yticks(ticks=[0, 1, 2], labels=["AAPL", "U.S. Economy", "NASDAQ"], rotation=0)
    plt.tight_layout()
    plt.savefig("data/outputs/sentiment_correlation_heatmap.png")
    plt.close()

    logging.info("All sentiment plots saved.")

def generate_summary(df, smoothing_window=3):
    """Generate a user-friendly summary with additional statistics."""
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    daily_sentiment = df.groupby(df["timestamp"].dt.date)["final_sentiment"].mean().reset_index()
    latest_date = daily_sentiment["timestamp"].max()
    latest_sentiment = daily_sentiment[daily_sentiment["timestamp"] == latest_date]["final_sentiment"].iloc[0]
    avg_sentiment = daily_sentiment["final_sentiment"].mean()
    # Calibrate thresholds
    low_threshold, high_threshold = calibrate_thresholds(df["final_sentiment"])
    # Interpret sentiment with calibrated thresholds
    sentiment_interpretation = "neutral"
    if latest_sentiment > high_threshold:
        sentiment_interpretation = "positive"
    elif latest_sentiment < low_threshold:
        sentiment_interpretation = "negative"
    # Compute distribution statistics
    sentiment_mean = df["final_sentiment"].mean()
    sentiment_median = df["final_sentiment"].median()
    sentiment_skew = skew(df["final_sentiment"].dropna())
    logging.info(f"Sentiment distribution stats - Mean: {sentiment_mean:.2f}, Median: {sentiment_median:.2f}, Skewness: {sentiment_skew:.2f}")
    # Write summary
    summary = (
        f"Sentiment Analysis Summary (as of {latest_date}):\n"
        f"We analyzed news about Apple (AAPL), the U.S. economy, and the NASDAQ market.\n"
        f"- Latest Sentiment: {latest_sentiment:.2f} ({sentiment_interpretation})\n"
        f"- Average Sentiment: {avg_sentiment:.2f}\n"
        f"- Sentiment Distribution Stats:\n"
        f"  - Mean: {sentiment_mean:.2f}\n"
        f"  - Median: {sentiment_median:.2f}\n"
        f"  - Skewness: {sentiment_skew:.2f}\n"
        f"What this means:\n"
        f"The sentiment score ranges from -1 (very negative) to 1 (very positive), with 0 being neutral.\n"
        f"Thresholds: Negative < {low_threshold:.2f}, Positive > {high_threshold:.2f}\n"
        f"A {sentiment_interpretation} sentiment suggests that recent news is "
        f"{'favorable' if sentiment_interpretation == 'positive' else 'unfavorable' if sentiment_interpretation == 'negative' else 'neutral'} "
        f"for Apple stock, the U.S. economy, and the NASDAQ market.\n"
        f"Check the plots in 'data/outputs/' for detailed visualizations:\n"
        f"- final_sentiment_plot.png: Overall sentiment trend with confidence intervals\n"
        f"- individual_sentiment_plot.png: Sentiment for AAPL, U.S. Economy, and NASDAQ\n"
        f"- sentiment_distribution_histogram.png: Distribution of sentiment scores\n"
        f"- sentiment_correlation_heatmap.png: Cross-source correlations"
    )
    with open("data/outputs/sentiment_summary.txt", "w") as f:
        f.write(summary)
    logging.info("Sentiment summary saved to data/outputs/sentiment_summary.txt")
    return summary

def validate_with_price(df):
    """Correlate sentiment with AAPL price returns."""
    try:
        aapl = yf.download("AAPL", start=df["timestamp"].min(), end=df["timestamp"].max())["Adj Close"]
        aapl = aapl.pct_change().dropna()
        aapl = aapl.reset_index()
        aapl["Date"] = aapl["Date"].dt.date
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        merged = df.groupby("date")["final_sentiment"].mean().reset_index().merge(aapl, left_on="date", right_on="Date")
        correlation = merged["final_sentiment"].corr(merged["Adj Close"])
        logging.info(f"Pearson correlation with AAPL returns: {correlation}")
        return correlation
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        return None

def main():
    config = load_config()
    # Load sentiment data with dynamic date
    current_date = datetime.now().strftime("%Y%m%d")
    stock_df = pd.read_csv(f"data/processed/stock_sentiment_{current_date}.csv")
    country_df = pd.read_csv(f"data/processed/country_sentiment_{current_date}.csv")
    market_df = pd.read_csv(f"data/processed/market_sentiment_{current_date}.csv")
    volatilities = pd.read_csv(f"data/raw/volatilities_{current_date}.csv")
    # Align DataFrames by timestamp with outer merge
    merged_df = stock_df[["timestamp", "sentiment_score", "days_elapsed"]].rename(columns={"sentiment_score": "stock"})
    merged_df = merged_df.merge(country_df[["timestamp", "sentiment_score"]].rename(columns={"sentiment_score": "country"}), 
                                on="timestamp", how="outer")
    merged_df = merged_df.merge(market_df[["timestamp", "sentiment_score"]].rename(columns={"sentiment_score": "market"}), 
                                on="timestamp", how="outer")
    if merged_df.empty:
        logging.error("Merged DataFrame is empty after outer merge.")
        raise ValueError("Empty merged DataFrame even with outer merge.")
    if "days_elapsed" not in merged_df.columns or merged_df["days_elapsed"].isna().all():
        merged_df["days_elapsed"] = stock_df["days_elapsed"].median() if not stock_df["days_elapsed"].isna().all() else 0
    logging.info(f"Merged sentiment data. Rows: {len(merged_df)}, Columns: {merged_df.columns.tolist()}")
    sentiment_data = merged_df[["timestamp", "stock", "country", "market", "days_elapsed"]]
    # Fit HSAM
    hsam = HSAM(
        decay_rate=config["model"]["decay_rate"],
        pca_components=config["model"]["pca_components"],
        bayesian_prior_strength=config["model"]["bayesian_prior_strength"],
        validation_split=config["model"].get("validation_split", 0.2)
    )
    result = hsam.fit(sentiment_data, volatilities)
    # Save and plot
    filepath = save_df(result, "outputs", f"final_sentiment_{current_date}")
    # Handle missing 'visualization' key in config
    visualization_config = config.get("visualization", {})
    smoothing_window = visualization_config.get("smoothing_window", 3)
    if "visualization" not in config:
        logging.warning("Configuration missing 'visualization' section. Using default smoothing_window=3. Please update config.yaml.")
    plot_sentiment(result, smoothing_window=smoothing_window)
    # Generate summary
    summary = generate_summary(result, smoothing_window=smoothing_window)
    print("\n=== Sentiment Summary ===\n")
    print(summary)
    # Validate with price data
    correlation = validate_with_price(result)
    logging.info(f"Latest sentiment score: {result['final_sentiment'].iloc[-1]}")
    return filepath

if __name__ == "__main__":
    main()