from constants import Constants as C

import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """Splits data into training, test and validation sets."""
    def __init__(self, test_size: float = 0.1, val_size: float = 0.1 / 0.9):
        self.test_size = test_size
        self.val_size = val_size
        self.label2id = None
        self.id2label = None

    def split_and_encode(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the DataFrame and adds numerical label IDs.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        unique_categories = df['category'].unique().tolist()

        # Create a mapping from category name to ID
        self.label2id = {label: i for i, label in enumerate(unique_categories)}
        self.id2label = {i: label for i, label in enumerate(unique_categories)}

        df['label_id'] = df['category'].map(self.label2id)
        logger.info("Label to ID mapping:", self.label2id)
        logger.info("ID to Label mapping:", self.id2label)

        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df['label_id'],
            random_state=C.SEED
        )

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.val_size,
            stratify=train_val_df['label_id'],
            random_state=C.SEED
        )

        logger.info(f"DataSplitter: Train set size: {len(train_df)}")
        logger.info(f"DataSplitter: Validation set size: {len(val_df)}")
        logger.info(f"DataSplitter: Test set size: {len(test_df)}")

        return train_df, val_df, test_df

    def get_mappings(self) -> tuple[dict, dict]:
        """Returns label2ID and ID2label mappings"""
        return self.label2id, self.id2label
