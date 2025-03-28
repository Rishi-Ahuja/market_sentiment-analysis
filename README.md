# Magnificent Seven Sentiment: Hierarchical Deep Learning for Multi-Source Financial News Analysis

## Overview

This project analyzes financial news sentiment for the "Magnificent Seven" tech stocks—Apple (AAPL), Microsoft (MSFT), Alphabet (GOOG), Amazon (AMZN), NVIDIA (NVDA), Tesla (TSLA), and Meta (META)—alongside U.S. economic and NASDAQ market trends. Using a custom **Hierarchical Sentiment Analysis Model (HSAM)**, it integrates data from Alpha Vantage and NewsAPI, processes it with FinBERT, and applies advanced techniques like PCA and Bayesian weighting. The result? Actionable sentiment scores, visualizations, and summaries to inform investment decisions.

## Description

This project delivers a powerful sentiment analysis tool for financial news, targeting the Magnificent Seven tech stocks—Apple (AAPL), Microsoft (MSFT), Alphabet (GOOG), Amazon (AMZN), NVIDIA (NVDA), Tesla (TSLA), and Meta (META). By integrating stock-specific news from Alpha Vantage and broader U.S. economic and NASDAQ market updates from NewsAPI, the system captures a holistic view of market sentiment. The HSAM pipeline processes this data through several stages: data collection with rate-limited API calls, text preprocessing to clean and filter news, sentiment scoring using the FinBERT model (yiyanghkust/finbert-tone), hierarchical modeling with exponential decay, PCA, and Bayesian weighting, and finally, visualization and reporting. Key features include temporal decay (older news weighted less), volatility adjustments, and dynamic weighting to balance stock, country, and market sentiments. Outputs include sentiment scores ranging from -1 to 1, time-series plots with confidence intervals, correlation heatmaps, and detailed summaries, all saved in the `data/outputs/` directory.

### Key Features
- **Multi-Source Data**: Combines stock-specific news, U.S. economy updates, and NASDAQ market insights.
- **Deep Learning**: Employs FinBERT (`yiyanghkust/finbert-tone`) for sentiment scoring.
- **HSAM Pipeline**: Features exponential decay, PCA, volatility adjustments, and dynamic Bayesian weighting.
- **Outputs**: Time-series plots, correlation heatmaps, all-stock trend comparisons, and text summaries.

## Prerequisites

- **Python 3.8+**
- **APIs**:
  - Alpha Vantage: Get your key [here](https://www.alphavantage.co/support/#api-key).
  - NewsAPI: Get your key [here](https://newsapi.org/register).
- **Hardware**: GPU recommended for FinBERT (optional; CPU works too).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Rishi-Ahuja/market_sentiment-analysis.git
   cd magnificent-seven-sentiment
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Sample `requirements.txt`:
   ```
   pandas
   numpy
   requests
   transformers
   torch
   scikit-learn
   matplotlib
   seaborn
   pyyaml
   ```

4. **Configure API Keys**:
   - Copy `config/config.yaml.example` to `config/config.yaml`.
   - Update with your API keys:
     ```yaml
     alpha_vantage:
       api_key: "YOUR_ALPHA_VANTAGE_KEY"
       base_url: "https://www.alphavantage.co/query"
     newsapi:
       api_key: "YOUR_NEWSAPI_KEY"
       base_url: "https://newsapi.org/v2/everything"
     ```

## Usage

1. **Run the Full Pipeline**:
   ```bash
   python pipeline.py
   ```
   - Fetches news, processes it, runs HSAM, and generates outputs in `data/outputs/`.

2. **Key Outputs**:
   - **CSV Files**: `data/outputs/final_sentiment_{ticker}.csv` (per stock) and `data/outputs/final_sentiment_combined.csv`.
   - **Plots**: PNGs like `sentiment_over_time_AAPL.png`, `all_stocks_sentiment_trends.png`.
   - **Summaries**: Text files like `sentiment_summary_AAPL.txt`.

3. **Customize**:
   - Edit `config/config.yaml` to adjust tickers, queries, or model parameters (e.g., `decay_rate`, `smoothing_window`).

## Project Structure

```
magnificent-seven-sentiment/
├── config/
│   ├── config.yaml          # API keys and settings
│   └── config.yaml.example  # Template config
├── data/
│   ├── raw/                # Raw API data
│   ├── processed/          # Preprocessed and sentiment-scored data
│   └── outputs/            # Final CSVs, plots, summaries
├── logs/                   # Log files (e.g., sentiment_20250327.log)
├── scripts/
│   ├── utils.py           # Helper functions (e.g., load_config, exponential_decay)
│   └── ...                # Other modular scripts
├── data_collection.py     # News fetching logic
├── preprocessing.py       # Text cleaning and prep
├── sentiment_analysis.py  # FinBERT sentiment scoring
├── model_training.py      # HSAM model and visualization
├── pipeline.py            # Main script to run everything
└── README.md              # This file
```

## How It Works

1. **Data Collection**: Fetches news for the Magnificent Seven, U.S. economy, and NASDAQ via APIs.
2. **Preprocessing**: Cleans text, filters stock-related content, and calculates days elapsed.
3. **Sentiment Analysis**: Scores text with FinBERT, adjusting confidence for positive/negative/neutral.
4. **HSAM**: Applies decay (0.1 rate), PCA (90% variance), Bayesian weighting, and volatility from `volatilities.csv`.
5. **Visualization**: Generates plots and summaries with a 3-day smoothing window.

## Sample Output

- **Sentiment Score (AAPL)**: -0.03 (positive trend as of March 27, 2025).
- **Plot**: `data/outputs/sentiment_over_time_AAPL.png` shows smoothed sentiment with 95% confidence intervals.

## Challenges

- **API Limits**: Alpha Vantage’s 5 calls/minute cap requires delays or synthetic data fallbacks.
- **Data Gaps**: No social media or full articles due to access constraints.

## Future Enhancements

- Integrate Reddit or financial reports.
- Optimize HSAM for larger datasets.
- Optimize the formula.

## Contributing

Feel free to fork, submit issues, or send pull requests!.

## License

MIT License—see [LICENSE](#).

## Acknowledgments

- Powered by Alpha Vantage, NewsAPI, and FinBERT.