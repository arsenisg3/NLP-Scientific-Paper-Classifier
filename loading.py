import os
import pandas as pd
from constants import Constants as C
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DataLoader:
    """Load dataset."""
    @staticmethod
    def load_data():
        try:
            code_directory = C.CURRENT_DIRECTORY
            logger.info(f"Loading data from {code_directory}...")
            dataset_directory = os.path.join(code_directory, 'dataset')
            df = pd.read_csv(dataset_directory + '/arXiv_scientific dataset.csv')
            logger.info(f"Data loaded successfully! Shape: {df.shape}")

        except FileNotFoundError as e:
            logger.error(f"Error: File not found - {e}")
            raise

        except Exception as e:
            logger.error(f"Error: An unexpected issue occurred while loading data - {e}")
            raise

        return df
