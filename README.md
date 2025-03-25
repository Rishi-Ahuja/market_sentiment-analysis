# Stock Market Sentiment Analysis

A comprehensive sentiment analysis system that analyzes news articles to predict market sentiment for stocks, with a focus on Apple (AAPL). The system combines multiple data sources including company-specific news, market trends, and economic indicators to provide nuanced sentiment scores.

## Features

- Multi-source sentiment analysis (Stock, Country, Market)
- Real-time data collection from Alpha Vantage, NewsAPI, and yfinance
- Advanced NLP using FinBERT model
- Comprehensive visualization suite
- Detailed statistical analysis and reporting

## Project Structure

```
sentiment-analysis-project/
├── config/              # Configuration files
├── data/               # Data directory
│   ├── raw/           # Raw data from APIs
│   ├── processed/     # Processed data
│   └── outputs/       # Final outputs and visualizations
├── logs/              # Log files
├── scripts/           # Python scripts
│   ├── data_collection.py    # Data collection from APIs
│   ├── preprocessing.py      # Data preprocessing
│   ├── sentiment_analysis.py # Sentiment analysis
│   ├── model_training.py     # Model training
│   ├── utils.py             # Utility functions
│   └── run_pipeline.py      # Main pipeline script
└── requirements.txt   # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd sentiment-analysis-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys in config/config.yaml:
```yaml
alpha_vantage_key: "your-key"
news_api_key: "your-key"
```

## Usage

Run the complete pipeline:
```bash
python scripts/run_pipeline.py
```

## Output

The system generates:
- Sentiment scores (-1 to 1)
- Visualization plots
- Statistical summaries
- Correlation analysis

## Dependencies

- Python 3.8+
- pandas
- numpy
- transformers (FinBERT)
- matplotlib
- seaborn
- yfinance
- requests

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
