from data_fields import SessionFields as SF

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import logging

matplotlib.use('Agg')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DataEDA:
    """
    Class used for EDA (both visualizing and printing).
    """
    def __init__(self, output_dir: str = './plots', savefig: bool = False):
        self.output_dir = output_dir
        self.savefig = savefig
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_category_distribution(self, df: pd.DataFrame, title: str, filename: str):
        """Plots the distribution of categories."""
        plt.figure(figsize=(12, 6))
        sns.countplot(y=df[SF.CATEGORY], order=df[SF.CATEGORY].value_counts().index, palette='viridis',
                      hue=df[SF.CATEGORY], legend=False)
        plt.title(title)
        plt.xlabel("Count")
        plt.ylabel("Category")
        plt.tight_layout()
        if self.savefig:
            plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        logger.debug(f"\n--- {title} Counts ---")
        logger.debug(df[SF.CATEGORY].value_counts())

    def plot_word_count_distribution(self, df: pd.DataFrame, title: str, filename: str):
        """Plots the distribution of word counts."""
        word_counts = df[SF.SUMMARY_WORD_COUNT]
        plt.figure(figsize=(10, 6))
        sns.histplot(word_counts, bins=50, kde=True)
        plt.title(title)
        plt.xlabel("Word Count")
        plt.ylabel("Count")
        plt.tight_layout()
        if self.savefig:
            plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        logger.debug(f"\n--- {title} Statistics ---")
        logger.debug(word_counts.describe())

    @staticmethod
    def show_duplicate_examples(df):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', None)

        logger.debug(df.head(5))
        logger.debug(df.describe())
        logger.debug(df.info())

        for col in ['category', 'first_author']:
            logger.debug(f"\n--- {col} unique values and counts ---")
            logger.debug(df[col].value_counts(dropna=False))

        logger.debug(f"Example title for duplicate summaries:")
        logger.debug(df[df[SF.SUMMARY] == df[df.duplicated(subset=[SF.SUMMARY])][SF.SUMMARY].iloc[0]][SF.TITLE])
        logger.debug(f"Example title and first author for duplicate titles:")
        logger.debug(df[df[SF.TITLE] == df[df.duplicated(subset=[SF.TITLE])][SF.TITLE].iloc[0]][[SF.TITLE, SF.FIRST_AUTHOR]])

        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')

    @staticmethod
    def show_nan_counts(df: pd.DataFrame, title: str = "NaN Counts"):
        """Prints NaN counts per column."""
        logger.debug(f"\n--- {title} ---")
        if df.isnull().sum().sum() == 0:
            logger.debug("No NaN values found.")
        else:
            logger.debug(df.isnull().sum())

    @staticmethod
    def compare_text_normalization(df: pd.DataFrame, num_examples: int = 3):
        """Shows side-by-side comparison of original and normalized text."""
        logger.debug("\n--- Original vs. Normalized Text Comparison ---")
        for i in range(min(num_examples, len(df[SF.SUMMARY]))):
            logger.debug(f"Example {i+1}:")
            logger.debug(f"Original: {df[SF.SUMMARY].iloc[i]}")
            logger.debug(f"Normalized: {df[SF.SUMMARY_NORM].iloc[i]}")
            logger.debug("-" * 30)

    def plot_and_save_confusion_matrix(self, trainer, test_dataset, model, custom_logger):
        """Plots the confusion matrix."""
        predictions_output = trainer.predict(test_dataset)
        predictions = np.argmax(predictions_output.predictions, axis=-1)
        true_labels = predictions_output.label_ids

        # Get class names from the model
        if hasattr(model.config, 'id2label') and model.config.id2label:
            class_names = [model.config.id2label[i] for i in sorted(model.config.id2label.keys())]
        else:
            custom_logger.warning("Model config does not have 'id2label'. Using numeric labels for confusion matrix.")
            class_names = [str(i) for i in range(model.config.num_labels)]

        # Calculate the confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=np.arange(model.config.num_labels))

        plt.figure(figsize=(len(class_names) + 2, len(class_names) + 2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=.5, linecolor='gray')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        if self.savefig:
            plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()
