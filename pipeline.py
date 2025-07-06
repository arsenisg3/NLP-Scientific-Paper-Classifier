import pandas as pd

import loading, cleaning, preprocess, model_loading, dataset_creation, training
from data_fields import SessionFields as SF
from constants import Constants as C
from eda import DataEDA
import logging
import torch
from datasets import Dataset
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MLPipeline:
    """Orchestrates the entire machine learning pipeline."""
    def __init__(self):
        self.df = None
        self.label2id = None
        self.id2label = None
        self.tokenizer = None
        self.model = None
        self.visualizer = DataEDA(output_dir=C.OUTPUT_PLOT_DIR, savefig=C.SAVEFIG)

    def run(self):
        logger.info("\n--- MLPipeline: Starting Data Loading & Preprocessing ---")
        self._load_and_preprocess_data()

        logger.info("\n--- MLPipeline: Splitting Data ---")
        train_df, val_df, test_df = self._split_data()

        logger.info("\n--- MLPipeline: Loading Model and Tokenizer ---")
        self._load_model_and_tokenizer()

        logger.info("\n--- MLPipeline: Creating Datasets ---")
        train_dataset_hf, val_dataset_hf, test_dataset_hf = self._create_datasets(train_df, val_df, test_df)

        logger.info("\n--- MLPipeline: Training Model ---")
        self._train_model(train_df, train_dataset_hf, val_dataset_hf, test_dataset_hf)

        logger.info("\n--- MLPipeline: Pipeline Complete! ---")

    def _load_and_preprocess_data(self):
        self.df = loading.DataLoader().load_data()
        logger.info(f"MLPipeline: Loaded data with {len(self.df)} rows.")

        self.visualizer.show_nan_counts(self.df, "Initial NaN Counts")
        self.visualizer.plot_word_count_distribution(self.df, "Initial Summary Lengths",
                                                     "initial_summary_lengths.png")
        self.visualizer.show_duplicate_examples(self.df)

        # Cleaning pipeline
        cleaning_tasks = [
            cleaning.DateFormatCleaner(time_cols=C.TIME_COLS, date_format=C.DATE_FORMAT),
            cleaning.TextNormalizationAndDeduplicationCleaner(),
            cleaning.NaSummaryRemover(),
            cleaning.JunkSummaryRemover(),
            cleaning.ShortWordSummaryRemover(),
            cleaning.LongWordSummaryRemover(),
            cleaning.ShortCategoryCountRemover(),
            #cleaning.MaxSamplesPerCategory(),  # uncomment to enable undersampling and hence reduce computational time
        ]

        for step in cleaning_tasks:
            logger.info(f"MLPipeline: Applying {step.__class__.__name__}...")
            self.df = step.apply(self.df)
            logger.info(f"MLPipeline: Current rows after {step.__class__.__name__}: {len(self.df)}")

        # Visualization / print statements
        self.visualizer.compare_text_normalization(self.df)
        self.visualizer.plot_category_distribution(self.df, "Category Distribution After Cleaning",
                                                   "final_category_distribution.png")
        self.visualizer.show_nan_counts(self.df, f"NaN Counts After Cleaning")
        self.visualizer.plot_word_count_distribution(self.df, "Final Summary Word Counts",
                                                     "final_summary_word_counts.png")

    def _split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        splitter = preprocess.DataSplitter()
        train_df, val_df, test_df = splitter.split_and_encode(self.df)
        self.label2id, self.id2label = splitter.get_mappings()
        return train_df, val_df, test_df

    def _load_model_and_tokenizer(self):
        model_loader = model_loading.ModelLoader(
            num_labels=len(self.label2id),
            label2id=self.label2id,
            id2label=self.id2label
        )
        self.tokenizer, self.model = model_loader.load()
        self.device = model_loader.move_to_device()

    def _create_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[Dataset, Dataset, Dataset]:
        dataset_creator = dataset_creation.DatasetCreator(
            tokenizer=self.tokenizer,
            text_column=SF.SUMMARY_NORM,
            label_column='label_id'
        )
        train_dataset_hf = dataset_creator.create_dataset(train_df)
        val_dataset_hf = dataset_creator.create_dataset(val_df)
        test_dataset_hf = dataset_creator.create_dataset(test_df)
        return train_dataset_hf, val_dataset_hf, test_dataset_hf

    def _train_model(self, train_df: pd.DataFrame, train_dataset_hf: Dataset, val_dataset_hf: Dataset, test_dataset_hf: Dataset):
        class_counts = train_df['label_id'].value_counts().sort_index()
        num_labels = self.model.config.num_labels
        num_samples = len(train_dataset_hf)
        class_weights_array = num_samples / (num_labels * class_counts.values)
        class_weights_tensor = torch.tensor(class_weights_array, dtype=torch.float)
        logger.info(f"Calculated class weights: {class_weights_tensor}")

        model_trainer = training.ModelTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset_hf,
            eval_dataset=val_dataset_hf,
            class_weights=class_weights_tensor
        )

        model_trainer.evaluate(test_dataset_hf)
        self.visualizer.plot_and_save_confusion_matrix(model_trainer, test_dataset_hf, self.model, logger)
