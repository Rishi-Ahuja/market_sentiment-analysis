import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=f"{log_dir}/sentiment_{datetime.now().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_config():
    """Load configuration from config.yaml."""
    try:
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)
        logging.info("Configuration loaded successfully.")
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise

def exponential_decay(weight, decay_rate, time_elapsed):
    """Apply exponential decay to weights with a minimum threshold."""
    return max(weight * np.exp(-decay_rate * time_elapsed), 0.01) if not pd.isna(weight) else 0

def market_impact(score, slope=0.5):
    """Convert raw sentiment score to final sentiment with tanh transformation."""
    return np.tanh(slope * score)

def save_df(df, folder, filename, ticker=None):
    """Save DataFrame to CSV with optional ticker suffix."""
    os.makedirs(f"data/{folder}", exist_ok=True)
    filepath = f"data/{folder}/{filename}{f'_{ticker}' if ticker else ''}.csv"
    df.to_csv(filepath, index=False)
    logging.info(f"Saved DataFrame to {filepath}")
    return filepath

def days_elapsed(timestamp):
    """Calculate days elapsed since timestamp."""
    current_date = datetime.now()
    try:
        pub_date = pd.to_datetime(timestamp)
        return (current_date - pub_date).days
    except Exception:
        return 0