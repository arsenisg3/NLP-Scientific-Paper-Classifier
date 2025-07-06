from abc import ABC, abstractmethod
import pandas as pd
import re
from data_fields import SessionFields as SF
from constants import Constants as C
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DataCleaner(ABC):
    """Abstract base class for data cleaning tasks."""
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the cleaning task to the DataFrame.
        Args:
            df (pd.DataFrame): The DataFrame to clean.
        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        pass


class DateFormatCleaner(DataCleaner):
    """Correct the format of date columns."""
    def __init__(self, time_cols: list, date_format: str):
        """
        Args:
            time_cols (list): List of time columns.
            date_format (str): The date format to use.
        """
        self.time_cols = time_cols
        self.date_format = date_format

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Correct the date format for each date column.

        Args:
            df (pd.DataFrame): Full DataFrame.

        Returns:
            pd.DataFrame: DataFrame with desired type for time columns.
        """
        df_copy = df.copy()
        for time_col in self.time_cols:
            df_copy[time_col] = pd.to_datetime(df_copy[time_col], format=self.date_format, errors='coerce').dt.normalize()

        return df_copy


class TextNormalizer:
    """A class to handle various text normalization operations in order to easier remove duplicates."""
    @staticmethod
    def normalize_string(text: str) -> str:
        """Applies a sequence of general text normalization steps."""
        if not isinstance(text, str):
            return text

        text = str(text).lower().strip()
        # Replace punctuation with a single space
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove multiple spaces and strip again
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def normalize_author_string(author_name: str) -> str:
        """Applies normalization specific to author names."""
        if not isinstance(author_name, str):
            return author_name

        author_name = str(author_name).lower().strip()
        author_name = author_name.replace('.', '')
        author_name = re.sub(r'\s+', ' ', author_name).strip()
        author_name = author_name.strip("'")
        split_name = author_name.split()
        # select only the first letter from each first/last name
        initials = [x[0] for x in split_name]
        new_author_name = ' '.join(initials)
        return new_author_name

    @staticmethod
    def normalize_summary(summary: str) -> str:
        summary = str(summary).lower().strip()
        summary = re.sub(r'[^\w\s]', ' ', summary)
        summary = re.sub(r'\s+', ' ', summary).strip()
        return summary


class TextNormalizationAndDeduplicationCleaner(DataCleaner):
    """
    A cleaner responsible for text normalization of specific columns
    and then removing duplicates based on those normalized columns.
    """
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()

        # Apply text normalization to each column
        df_cleaned[SF.TITLE_NORM] = df_cleaned[SF.TITLE].apply(TextNormalizer.normalize_string)
        df_cleaned[SF.FIRST_AUTHOR_NORM] = df_cleaned[SF.FIRST_AUTHOR].apply(TextNormalizer.normalize_author_string)
        df_cleaned[SF.SUMMARY_NORM] = df_cleaned[SF.SUMMARY].apply(TextNormalizer.normalize_summary)

        # Remove duplicates based on the normalized title and first author columns
        subset_cols_for_dedup = [SF.TITLE_NORM, SF.FIRST_AUTHOR_NORM]
        df_deduplicated = df_cleaned.drop_duplicates(subset=subset_cols_for_dedup, keep='first')
        # Same with normalized summary column
        df_deduplicated2 = df_deduplicated.drop_duplicates(subset=SF.SUMMARY_NORM, keep='first')

        df_post_deduplication = df_deduplicated2.drop(columns=[SF.TITLE_NORM, SF.FIRST_AUTHOR_NORM])

        return df_post_deduplication


class NaSummaryRemover(DataCleaner):
    """Removes rows where summary is Nan."""
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()

        df_nan_removed = df_copy[df_copy[SF.SUMMARY].notna()]
        return df_nan_removed


class JunkSummaryRemover(DataCleaner):
    """Removes rows where the normalized summary matches a predefined list of junk strings."""
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        is_junk = df_copy[SF.SUMMARY_NORM].apply(lambda x: x in C.JUNK_VALUES if pd.notna(x) else False)
        df_junk_removed = df_copy[~is_junk]
        return df_junk_removed


class ShortWordSummaryRemover(DataCleaner):
    """Removes rows where the normalized summary has fewer words than a minimum length."""
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        is_short_word = df_copy[SF.SUMMARY_NORM].apply(
            lambda x: len(str(x).split()) < C.MIN_WORDS if pd.notna(x) else True
        )
        df_cleaned = df_copy[~is_short_word]
        return df_cleaned


class LongWordSummaryRemover(DataCleaner):
    """Removes rows where the normalized summary has more words than a maximum length."""
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        is_short_word = df_copy[SF.SUMMARY_NORM].apply(
            lambda x: len(str(x).split()) > C.MAX_WORDS if pd.notna(x) else True
        )
        df_cleaned = df_copy[~is_short_word]
        return df_cleaned


class ShortCategoryCountRemover(DataCleaner):
    """Removes categories having a number of values less than a minimum amount."""
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        desired_category_count_threshold = C.DESIRED_CATEGORY_COUNT_THRESHOLD
        category_counts = df['category'].value_counts()
        filtered_categories_series = category_counts[category_counts > desired_category_count_threshold]
        desired_category_names = filtered_categories_series.index.tolist()

        df_updated = df[df['category'].isin(desired_category_names)].copy()
        unique_categories = df_updated['category'].unique()
        logger.info(f"Post cleaning dataset contains {len(unique_categories)} unique categories.")
        return df_updated


class MaxSamplesPerCategory(DataCleaner):
    """Select a certain number of samples for each category (undersampling approach for running speed purposes)."""
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        sampled_dfs = []
        desired_max_samples_per_category = C.MAX_SAMPLES_PER_CATEGORY

        for category_name in df['category'].unique():
            category_df = df[df['category'] == category_name]

            if len(category_df) > desired_max_samples_per_category:
                sampled_category_df = category_df.sample(n=desired_max_samples_per_category, random_state=C.SEED)
            else:
                sampled_category_df = category_df

            sampled_dfs.append(sampled_category_df)

        # Concatenate all sampled (or fully retained) categories back into a single DataFrame
        df_sampled = pd.concat(sampled_dfs).sample(frac=1, random_state=C.SEED).reset_index(drop=True)

        logger.info("\n--- Final Sampled DataFrame Category Counts ---")
        logger.info(df_sampled['category'].value_counts())
        logger.info(f"\nTotal rows after sampling: {len(df_sampled)}")
        return df_sampled
