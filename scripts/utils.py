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
    """Apply exponential decay to weights."""
    return weight * np.exp(-decay_rate * time_elapsed)

def market_impact(score, slope=0.5):
    """
    Convert raw sentiment score to final sentiment with linear scaling and soft clipping.
    Returns values between -1 and 1 with more uniform distribution.
    """
    # Initial linear scaling
    scaled = score * slope
    # Soft clipping using arctangent
    return np.arctan(scaled) / (np.pi/2)

def save_df(df, folder, filename):
    """Save DataFrame to CSV with timestamp."""
    os.makedirs(f"data/{folder}", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    filepath = f"data/{folder}/{filename}_{timestamp}.csv"
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
        return 0  # Default to 0 if parsing fails