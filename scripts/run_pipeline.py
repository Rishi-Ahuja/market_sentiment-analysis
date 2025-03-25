import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data_collection import main as data_collection_main
from preprocessing import main as preprocessing_main
from sentiment_analysis import main as sentiment_analysis_main
from model_training import main as model_training_main
from utils import logging

def run_pipeline():
    """Run the full HSAM pipeline."""
    try:
        logging.info("Starting pipeline execution.")
        data_collection_main()
        preprocessing_main()
        sentiment_analysis_main()
        final_csv = model_training_main()
        logging.info("Pipeline completed successfully.")
        print(f"Final sentiment scores saved to: {final_csv}")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()