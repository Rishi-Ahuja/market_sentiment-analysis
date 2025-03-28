import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal, gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scripts.utils import load_config, save_df, logging, exponential_decay, market_impact
import os
import matplotlib.dates as mdates

# Set seaborn style for better aesthetics
plt.style.use('seaborn')
sns.set_palette("deep")

class HSAM:
    def __init__(self, decay_rate=0.1, min_signal=0.01, explained_variance=0.9, prior_strength=0.1, validation_split=0.2):
        self.decay_rate = decay_rate
        self.min_signal = min_signal
        self.pca = PCA()
        self.explained_variance = explained_variance
        self.prior_strength = prior_strength
        self.weights = None
        self.validation_split = validation_split

    def fit(self, sentiment_data, volatilities, ticker=None):
        train_size = int(len(sentiment_data) * (1 - self.validation_split))
        train_data = sentiment_data.iloc[:train_size].copy()
        val_data = sentiment_data.iloc[train_size:].copy()
        logging.info(f"Training ({ticker or 'combined'}): {len(train_data)} rows, Validation: {len(val_data)} rows")

        train_result = self._fit_subset(train_data, volatilities, "train", ticker)
        val_result = self._fit_subset(val_data, volatilities, "val", ticker) if len(val_data) >= 1 else pd.DataFrame()
        result = pd.concat([train_result, val_result], ignore_index=True) if not val_result.empty else train_result
        return result

    def _fit_subset(self, data, volatilities, subset_name="train", ticker=None):
        data["days_elapsed"] = data["days_elapsed"].fillna(data["days_elapsed"].median() or 0)
        sentiment_cols = ["stock", "country", "market"]
        valid_rows = data[sentiment_cols].notna().sum(axis=1) >= 1
        if not valid_rows.any() or len(data[valid_rows]) < 1:
            logging.warning(f"No sufficient sentiment data in {subset_name} subset for {ticker or 'combined'}. Using all rows with defaults.")
            data = data.copy()
            data[sentiment_cols] = data[sentiment_cols].fillna(0)
        else:
            data = data[valid_rows].copy()
        logging.info(f"{subset_name.capitalize()} rows after filter ({ticker or 'combined'}): {len(data)}")

        # Apply decay, preserving original values
        data[sentiment_cols] = data.apply(
            lambda row: [exponential_decay(row[col], self.decay_rate, row["days_elapsed"])
                         if pd.notna(row[col]) and abs(row[col]) > self.min_signal else row[col] if pd.notna(row[col]) else 0
                         for col in sentiment_cols], axis=1, result_type="expand"
        )

        imputer = SimpleImputer(strategy="constant", fill_value=0)
        sentiment_matrix = imputer.fit_transform(data[sentiment_cols])
        
        # Standardize only country and market, preserve stock sentiment
        stock_sentiment = sentiment_matrix[:, 0].copy()  # Preserve stock sentiment
        country_market = sentiment_matrix[:, 1:]  # Country and market
        scaler = StandardScaler()
        country_market_scaled = scaler.fit_transform(country_market)
        sentiment_matrix = np.column_stack((stock_sentiment, country_market_scaled))
        sentiment_matrix = np.nan_to_num(sentiment_matrix, nan=0, posinf=0, neginf=0)

        if sentiment_matrix.shape[0] <= 1:
            logging.warning(f"Too few rows ({sentiment_matrix.shape[0]}) for PCA in {subset_name} ({ticker or 'combined'}). Returning raw sentiment.")
            return pd.DataFrame({
                "timestamp": data["timestamp"],
                "stock": data["stock"],
                "country": data["country"],
                "market": data["market"],
                "final_sentiment": data["stock"].fillna(0),
                "ticker": ticker if ticker else data.get("ticker", "combined")
            })

        # PCA with sign alignment to stock sentiment
        pca_features = self.pca.fit_transform(sentiment_matrix)
        n_components = min(np.argmax(np.cumsum(self.pca.explained_variance_ratio_) >= self.explained_variance) + 1, sentiment_matrix.shape[1])
        pca_features = pca_features[:, :n_components]
        # Align first component with stock sentiment
        stock_col = sentiment_matrix[:, 0]  # Stock is first column
        if np.corrcoef(pca_features[:, 0], stock_col)[0, 1] < 0:
            pca_features[:, 0] *= -1
        s_pca = pd.DataFrame(pca_features, columns=[f"P{i+1}" for i in range(n_components)])

        # Volatility weights, capped to prevent dominance
        ticker_vol = volatilities[volatilities["ticker"] == ticker]["volatility"].iloc[0] if ticker in volatilities["ticker"].values else 0.02
        vol_weights = np.array([min(ticker_vol, 0.5), 0.2, 0.2])  # Stock, country, market
        if len(vol_weights) < n_components:
            vol_weights = np.pad(vol_weights, (0, n_components - len(vol_weights)), mode='constant', constant_values=0.02)
        self.weights = vol_weights[:n_components] / (vol_weights[:n_components].sum() or 1)

        # Dynamic weighting with Bayesian update
        weights_dynamic = []
        cov = np.cov(pca_features.T) + np.eye(n_components) * 1e-6
        cov = np.nan_to_num(cov, nan=0, posinf=0, neginf=0)

        for row in s_pca.values:
            likelihood = multivariate_normal.pdf(row, mean=np.zeros(n_components), cov=cov, allow_singular=True)
            prior = multivariate_normal.pdf(self.weights, mean=np.full(n_components, 0.5), cov=np.eye(n_components) * self.prior_strength)
            evidence = likelihood * prior + 1e-10
            self.weights = np.clip((self.weights * prior + row * likelihood) / evidence, -1, 1)
            weights_dynamic.append(self.weights.copy())

        # Weighted scores with stronger baseline from raw stock sentiment
        raw_stock_mean = data["stock"].mean()
        pca_contribution = [np.dot(row, w) for row, w in zip(s_pca.values, weights_dynamic)]
        # Scale PCA contribution to avoid overpowering the raw stock mean
        pca_std = np.std(pca_contribution)
        scaled_pca_contribution = [x / (pca_std or 1) * 0.5 for x in pca_contribution]  # Reduce PCA impact
        weighted_scores = [raw_stock_mean + pca_contrib for pca_contrib in scaled_pca_contribution]
        
        # Shift the scores to center around 0
        score_mean = np.mean(weighted_scores)
        weighted_scores = [score - score_mean for score in weighted_scores]
        
        # Normalize with a larger scaling factor to spread out the scores
        score_range = np.ptp(weighted_scores) or 1  # Range of weighted scores
        final_sentiment = [market_impact(score / score_range * 6) for score in weighted_scores]  # Scale to [-6, 6] before market_impact

        return pd.DataFrame({
            "timestamp": data["timestamp"],
            "stock": data["stock"],
            "country": data["country"],
            "market": data["market"],
            "final_sentiment": final_sentiment,
            "ticker": ticker if ticker else data.get("ticker", "combined")
        })

def compute_confidence_intervals(df, n_bootstrap=1000, ci_level=0.95):
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily_groups = df.groupby("date")
    ci_lower, ci_upper = [], []
    block_size = 5
    for date, group in daily_groups:
        scores = group["final_sentiment"].values
        bootstrapped_means = []
        for _ in range(n_bootstrap):
            start_idx = np.random.randint(0, max(1, len(scores) - block_size + 1))
            sample = scores[start_idx:start_idx + block_size] if len(scores) >= block_size else scores
            bootstrapped_means.append(np.mean(np.random.choice(sample, len(scores), replace=True)))
        ci = np.percentile(bootstrapped_means, [(1 - ci_level) / 2 * 100, (1 + ci_level) / 2 * 100])
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])
    daily_sentiment = daily_groups["final_sentiment"].mean().reset_index()
    daily_sentiment["timestamp"] = pd.to_datetime(daily_sentiment["date"])
    return daily_sentiment.sort_values("timestamp"), ci_lower, ci_upper

def calibrate_thresholds(final_sentiment):
    # Use 25th and 75th percentiles for thresholds
    low = np.percentile(final_sentiment, 25)
    high = np.percentile(final_sentiment, 75)
    logging.info(f"Percentile-based thresholds: Negative < {low:.2f}, Positive > {high:.2f}")
    return low, high

def plot_sentiment(df, smoothing_window=3, ticker=None):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    raw_sentiments = df[["timestamp", "stock", "country", "market"]].copy()
    title_suffix = f" ({ticker})" if ticker else " (Magnificent 7 Combined)"
    output_dir = "data/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Compute percentile-based thresholds for shading the neutral zone
    low_threshold, high_threshold = calibrate_thresholds(df["final_sentiment"])

    # Plot 1: Sentiment Over Time
    daily_sentiment, ci_lower, ci_upper = compute_confidence_intervals(df)
    daily_sentiment["smoothed"] = daily_sentiment["final_sentiment"].rolling(window=smoothing_window, min_periods=1).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(daily_sentiment["timestamp"], daily_sentiment["final_sentiment"], label="Daily Sentiment", alpha=0.3, color="gray")
    plt.plot(daily_sentiment["timestamp"], daily_sentiment["smoothed"], label=f"Smoothed ({smoothing_window}-day)", linewidth=3, color="navy")
    plt.fill_between(daily_sentiment["timestamp"], ci_lower, ci_upper, color="navy", alpha=0.15, label="95% Confidence Interval")
    plt.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    # Shade the neutral zone
    plt.fill_between(daily_sentiment["timestamp"], low_threshold, high_threshold, color="gray", alpha=0.1, label="Neutral Zone")
    plt.title(f"Sentiment Over Time{title_suffix}", fontsize=16, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sentiment Score", fontsize=12)
    plt.legend(fontsize=10, loc="best")
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_over_time{'_' + ticker if ticker else ''}.png", dpi=300)
    plt.close()

    # Plot 2: Individual Sentiment Trends
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("deep", 3)  # Distinct colors for stock, country, market
    for col, label, color in zip(["stock", "country", "market"], ["Stock", "U.S. Economy", "NASDAQ"], colors):
        daily = raw_sentiments.groupby(raw_sentiments["timestamp"].dt.date)[col].mean().reset_index()
        daily["timestamp"] = pd.to_datetime(daily["timestamp"])
        plt.plot(daily["timestamp"], daily[col], label=label, color=color, linewidth=3)
    plt.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    # Shade the neutral zone
    plt.fill_between(daily["timestamp"], low_threshold, high_threshold, color="gray", alpha=0.1, label="Neutral Zone")
    plt.title(f"Individual Sentiment Trends{title_suffix}", fontsize=16, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sentiment Score", fontsize=12)
    plt.legend(fontsize=10, loc="best")
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/individual_sentiment_trends{'_' + ticker if ticker else ''}.png", dpi=300)
    plt.close()

    # Plot 3: Correlation Heatmap
    corr_matrix = raw_sentiments[["stock", "country", "market"]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, center=0, 
                annot_kws={"size": 12}, cbar_kws={"label": "Correlation Coefficient"},
                linewidths=0.5, linecolor="black")
    plt.title(f"Correlation Between Sentiment Sources{title_suffix}", fontsize=16, pad=15)
    plt.xticks(ticks=range(3), labels=["Stock", "U.S. Economy", "NASDAQ"], rotation=45, fontsize=12)
    plt.yticks(ticks=range(3), labels=["Stock", "U.S. Economy", "NASDAQ"], rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_correlation_heatmap{'_' + ticker if ticker else ''}.png", dpi=300)
    plt.close()

    logging.info(f"Static plots saved as PNGs for {ticker or 'combined'}.")

def plot_all_stocks(results, smoothing_window=3):
    output_dir = "data/outputs"
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(14, 8))

    # Compute percentile-based thresholds for the combined data
    combined_df = pd.concat([df for df in results.values()], ignore_index=True)
    low_threshold, high_threshold = calibrate_thresholds(combined_df["final_sentiment"])

    # Use a distinct color palette for different stocks
    colors = sns.color_palette("tab10", len(results))  # More distinct colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>']  # Different markers for each stock
    for (ticker, df), color, marker in zip(results.items(), colors, markers * (len(results) // len(markers) + 1)):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        daily_sentiment = df.groupby(df["timestamp"].dt.date)["final_sentiment"].mean().reset_index()
        daily_sentiment["timestamp"] = pd.to_datetime(daily_sentiment["timestamp"])
        daily_sentiment["smoothed"] = daily_sentiment["final_sentiment"].rolling(window=smoothing_window, min_periods=1).mean()
        plt.plot(daily_sentiment["timestamp"], daily_sentiment["smoothed"], label=ticker, linewidth=2, color=color, marker=marker, markevery=5, markersize=6)

    plt.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    # Shade the neutral zone
    plt.fill_between(daily_sentiment["timestamp"], low_threshold, high_threshold, color="gray", alpha=0.1, label="Neutral Zone")
    plt.title("Sentiment Trends for All Stocks (Smoothed)", fontsize=16, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sentiment Score", fontsize=12)
    # Improved legend with semi-transparent background
    legend = plt.legend(title="Stocks", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10, title_fontsize=12)
    legend.get_frame().set_alpha(0.9)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/all_stocks_sentiment_trends.png", dpi=300)
    plt.close()
    logging.info("All stocks sentiment plot saved as 'all_stocks_sentiment_trends.png'.")

def generate_summary(df, smoothing_window=3, ticker=None):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    daily_sentiment = df.groupby(df["timestamp"].dt.date)["final_sentiment"].mean().reset_index()
    latest_date = daily_sentiment["timestamp"].max()
    latest_sentiment = daily_sentiment[daily_sentiment["timestamp"] == latest_date]["final_sentiment"].iloc[0]
    avg_sentiment = daily_sentiment["final_sentiment"].mean()
    low_threshold, high_threshold = calibrate_thresholds(df["final_sentiment"])
    
    # Labeling logic with strict inequalities
    if latest_sentiment < low_threshold:
        sentiment_label = "negative"
    elif latest_sentiment > high_threshold:
        sentiment_label = "positive"
    else:
        sentiment_label = "neutral"
    
    stats = {
        "mean": df["final_sentiment"].mean(),
        "median": df["final_sentiment"].median(),
        "min": df["final_sentiment"].min(),
        "max": df["final_sentiment"].max()
    }
    trend = "stable"
    smoothed = daily_sentiment["final_sentiment"].rolling(window=smoothing_window, min_periods=1).mean()
    if len(smoothed) > smoothing_window:
        trend_slope = (smoothed.iloc[-1] - smoothed.iloc[-smoothing_window]) / smoothing_window
        trend = "rising" if trend_slope > 0.01 else "falling" if trend_slope < -0.01 else "stable"

    # Simplified summary
    summary = (
        f"Sentiment Summary for {ticker or 'Magnificent 7 Combined'} (as of {latest_date.strftime('%Y-%m-%d')}):\n"
        f"\nWhat’s Happening Now:\n"
        f"- Latest Sentiment: {latest_sentiment:.2f} ({sentiment_label.capitalize()})\n"
        f"  This is the sentiment score for the most recent day. "
        f"{'It’s positive, meaning recent news is generally good.' if sentiment_label == 'positive' else 'It’s negative, meaning recent news is generally bad.' if sentiment_label == 'negative' else 'It’s neutral, meaning recent news is balanced.'}\n"
        f"- Average Sentiment: {avg_sentiment:.2f}\n"
        f"  This is the average sentiment over all days. It shows the overall mood of the news over time.\n"
        f"- Trend (Last {smoothing_window} Days): {trend.capitalize()}\n"
        f"  This shows if the sentiment is getting better ('rising'), worse ('falling'), or staying the same ('stable') over the last {smoothing_window} days.\n"
        f"\nSentiment Categories:\n"
        f"- Negative: Below {low_threshold:.2f}\n"
        f"- Positive: Above {high_threshold:.2f}\n"
        f"  Scores in between are considered neutral. These ranges help us label the sentiment as positive, neutral, or negative.\n"
        f"\nKey Stats:\n"
        f"- Overall Average: {stats['mean']:.2f}\n"
        f"- Middle Value (Median): {stats['median']:.2f}\n"
        f"- Range: {stats['min']:.2f} to {stats['max']:.2f}\n"
        f"  These numbers show the typical sentiment score, the middle score, and the full range of scores we’ve seen.\n"
        f"\nWhat This Means:\n"
        f"The latest news for {ticker or 'the Magnificent 7 stocks'} is {sentiment_label}. "
        f"Since the trend is {trend}, the sentiment might {'improve' if trend == 'rising' else 'worsen' if trend == 'falling' else 'stay about the same'} in the coming days. "
        f"Check the graphs in data/outputs/ to see the full picture!"
    )
    with open(f"data/outputs/sentiment_summary{'_' + ticker if ticker else ''}.txt", "w") as f:
        f.write(summary)
    logging.info(f"Summary saved for {ticker or 'combined'}.")
    return summary

def main():
    config = load_config()
    volatilities = pd.read_csv("data/raw/volatilities.csv")
    hsam = HSAM(
        decay_rate=config["model"].get("decay_rate", 0.1),
        min_signal=config["model"].get("min_signal", 0.01),
        explained_variance=config["model"].get("explained_variance", 0.9),
        prior_strength=config["model"].get("bayesian_prior_strength", 0.1),
        validation_split=config["model"].get("validation_split", 0.2)
    )
    smoothing_window = config.get("visualization", {}).get("smoothing_window", 3)

    results = {}
    for ticker in config["tickers"]["stocks"]:
        try:
            stock_df = pd.read_csv(f"data/processed/stock_sentiment_{ticker}.csv")
            country_df = pd.read_csv("data/processed/country_sentiment.csv")
            market_df = pd.read_csv("data/processed/market_sentiment.csv")

            # Keep exact timestamps for stock news
            stock_df["timestamp"] = pd.to_datetime(stock_df["timestamp"])
            stock_df["date"] = stock_df["timestamp"].dt.date
            stock_df = stock_df.rename(columns={"sentiment_score": "stock"})

            # Aggregate country and market to daily level
            country_df["date"] = pd.to_datetime(country_df["timestamp"]).dt.date
            market_df["date"] = pd.to_datetime(market_df["timestamp"]).dt.date

            daily_country = country_df.groupby("date")["sentiment_score"].mean().reset_index().rename(columns={"sentiment_score": "country"})
            daily_market = market_df.groupby("date")["sentiment_score"].mean().reset_index().rename(columns={"sentiment_score": "market"})

            # Merge country and market daily data with stock data on date
            merged_df = stock_df.merge(daily_country, on="date", how="left").merge(daily_market, on="date", how="left")

            # Add days_elapsed from stock_df
            merged_df["days_elapsed"] = stock_df["days_elapsed"]

            # Fill NaNs with overall mean sentiment for country and market
            merged_df["country"] = merged_df["country"].fillna(country_df["sentiment_score"].mean())
            merged_df["market"] = merged_df["market"].fillna(market_df["sentiment_score"].mean())

            if merged_df.empty:
                logging.error(f"Merged DataFrame empty for {ticker}.")
                continue
            
            result = hsam.fit(merged_df, volatilities, ticker)
            results[ticker] = result
            save_df(result, "outputs", "final_sentiment", ticker)
            plot_sentiment(result, smoothing_window, ticker)
            summary = generate_summary(result, smoothing_window, ticker)
            print(f"\n=== Sentiment Summary for {ticker} ===\n{summary}")
        except Exception as e:
            logging.error(f"Failed to process {ticker} in model training: {e}")
            continue

    combined_df = pd.concat(results.values(), ignore_index=True)
    save_df(combined_df, "outputs", "final_sentiment", "combined")
    plot_sentiment(combined_df, smoothing_window)
    plot_all_stocks(results, smoothing_window)
    summary = generate_summary(combined_df, smoothing_window)
    print(f"\n=== Combined Sentiment Summary ===\n{summary}")
    logging.info("Pipeline completed successfully.")
    return "data/outputs/final_sentiment_combined.csv"

if __name__ == "__main__":
    main()